import numpy as np


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.ndim == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.max <= 255:
            return arr.astype(np.uint8, copy=False)
        scaled = (arr.astype(np.float32) - info.min) * (255.0 / max(1, info.max - info.min))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    arr_f = arr.astype(np.float32)
    bands = min(arr_f.shape[2], 3)
    out = np.zeros((arr_f.shape[0], arr_f.shape[1], 3), dtype=np.uint8)

    for i in range(bands):
        ch = arr_f[:, :, i]
        nz = ch[ch > 0]
        if nz.size > 0:
            lo, hi = np.percentile(nz, [2, 98])
            if hi <= lo:
                lo, hi = float(ch.min()), float(ch.max())
        else:
            lo, hi = 0.0, 1.0
        if hi > lo:
            out[:, :, i] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if bands < 3:
        out[:, :, 1] = out[:, :, 0]
        out[:, :, 2] = out[:, :, 0]

    return out
