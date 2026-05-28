from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.image_utils import to_uint8
from utils.window import gaussian_window


def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    num_classes: int,
    transform: Callable | None = None,
    batch_size: int = 16,
    device: torch.device | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    tta_fn: Callable | None = None,
) -> np.ndarray:
    H, W = image.shape[:2]
    if H == 0 or W == 0:
        raise ValueError(f"Invalid image dimensions: {H}x{W}")
    if patch_size <= 0:
        raise ValueError(f"Invalid patch_size: {patch_size}")
    if num_classes <= 0:
        raise ValueError(f"Invalid num_classes: {num_classes}")

    stride = max(1, patch_size - overlap)
    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    window = gaussian_window(patch_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_inputs: list = []
    batch_coords: list = []

    with torch.no_grad():
        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r2, c2 = min(r + patch_size, H), min(c + patch_size, W)
                if r2 <= r or c2 <= c:
                    continue
                patch = image[r:r2, c:c2].copy()
                if patch.size == 0:
                    continue
                ph, pw = patch.shape[:2]

                if ph < patch_size or pw < patch_size:
                    pad = cv2.copyMakeBorder(
                        patch, 0, patch_size - ph, 0, patch_size - pw, cv2.BORDER_REFLECT_101
                    )
                else:
                    pad = patch

                aug = transform(image=pad) if transform else {"image": torch.from_numpy(to_uint8(pad).transpose(2, 0, 1))}
                inp = aug["image"]

                batch_inputs.append(inp)
                batch_coords.append((r, r2, c, c2, ph, pw))

                if len(batch_inputs) == batch_size:
                    _run_batch(model, batch_inputs, batch_coords, prob_sum, count_map,
                               window, device, amp_dtype, tta_fn, num_classes)
                    batch_inputs = []
                    batch_coords = []

        if batch_inputs:
            _run_batch(model, batch_inputs, batch_coords, prob_sum, count_map,
                       window, device, amp_dtype, tta_fn, num_classes)

    averaged = prob_sum / np.maximum(count_map, 1e-6)
    return averaged


def _run_batch(
    model: torch.nn.Module,
    inputs: list,
    coords: list[tuple[int, int, int, int, int, int]],
    prob_sum: np.ndarray,
    count_map: np.ndarray,
    window: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    tta_fn: Callable | None,
    num_classes: int,
) -> None:
    inp_tensor = torch.stack(inputs).to(device)
    if tta_fn is not None:
        probs = tta_fn(model, inp_tensor, num_classes, amp_dtype).cpu().numpy()
    else:
        with torch.amp.autocast(device.type if hasattr(device, 'type') else str(device), dtype=amp_dtype):
            raw = model(inp_tensor)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            probs = torch.softmax(raw.float(), 1).cpu().numpy()
    for i, (br, br2, bc, bc2, bph, bpw) in enumerate(coords):
        wind_slice = window[:bph, :bpw]
        prob_sum[:, br:br2, bc:bc2] += probs[i, :, :bph, :bpw] * wind_slice
        count_map[br:br2, bc:bc2] += wind_slice
