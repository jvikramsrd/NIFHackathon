"""
data/dataset.py  (A4000-optimised)
────────────────────────────────────
• Larger patch/crop sizes matching new config
• Extra augmentations for drone aerial imagery
• Windows-safe: all DataLoader logic sets num_workers properly
• Persistent workers + prefetch_factor for 10-worker setup
"""

import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Segmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────


class SegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        patch_size=640,
        patch_sizes=None,
        num_classes=4,
        transform=None,
        is_train=True,
        file_list=None,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes
        self.patch_sizes = tuple(patch_sizes or (patch_size,))
        if file_list is not None:
            self.img_paths = [Path(p) for p in file_list if Path(p).exists()]
        else:
            self.img_paths = [p for p in self.img_dir.glob("*.png") if p.stat().st_size > 0]
        if len(self.img_paths) == 0:
            raise ValueError(f"No valid PNG patches in {img_dir}")
        self.transform = transform or (
            get_train_transforms(patch_size, self.patch_sizes)
            if is_train
            else get_val_transforms(patch_size)
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        img = cv2.imread(str(ip))
        if img is None:
            raise ValueError(f"Failed to read image at {ip}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp = self.mask_dir / ip.name
        if mp.exists() and mp.stat().st_size > 0:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = np.clip(mask, 0, self.num_classes - 1)

        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A — Rooftop Classification Dataset
# ─────────────────────────────────────────────────────────────────────────────


class RooftopDataset(Dataset):
    def __init__(
        self,
        root_dir,
        class_names,
        crop_size=224,
        transform=None,
        is_train=True,
        samples=None,
        use_randaugment=False,
        randaugment_n=2,
        randaugment_m=7,
    ):
        self.root = Path(root_dir)
        self.class_names = class_names
        self.c2id = {c: i for i, c in enumerate(class_names)}

        if samples is not None:
            self.samples = samples
        else:
            self.samples: List[Tuple[Path, int]] = []
            for cls in class_names:
                cls_dir = self.root / cls
                if cls_dir.is_dir():
                    for p in cls_dir.glob("*.png"):
                        if p.stat().st_size > 0:
                            self.samples.append((p, self.c2id[cls]))
            self.samples.sort(key=lambda x: str(x[0]))

        if len(self.samples) == 0:
            raise ValueError(f"No valid crops in {root_dir}")

        self.transform = transform or (
            get_clf_train_transforms(
                crop_size,
                use_randaugment=use_randaugment,
                randaugment_n=randaugment_n,
                randaugment_m=randaugment_m,
            )
            if is_train
            else get_clf_val_transforms(crop_size)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            img = np.zeros((self.transform.transforms[0].size[0] if hasattr(self.transform.transforms[0], 'size') else 224,
                            self.transform.transforms[0].size[1] if hasattr(self.transform.transforms[0], 'size') else 224, 3), dtype=np.uint8)

        img_f = img.astype(np.float32)
        mean = img_f.mean(axis=(0, 1), keepdims=True)
        std = img_f.std(axis=(0, 1), keepdims=True) + 1e-6
        img_norm = (img_f - mean) / std * 64.0 + 127.0
        img = np.clip(img_norm, 0, 255).astype(np.uint8)

        img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)

    def class_weights(self):
        labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)
        counts = torch.from_numpy(
            np.bincount(labels, minlength=len(self.class_names)).astype(np.float32)
        )
        w = 1.0 / (counts + 1e-6)
        return w / w.sum() * len(self.class_names)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# All tuned for drone aerial imagery (top-down perspective, no horizon)
# ─────────────────────────────────────────────────────────────────────────────


def get_train_transforms(patch_size: int = 640, patch_sizes=None):
    patch_sizes = tuple(int(p) for p in (patch_sizes or (patch_size,)))
    multi_scale_crops = []
    for crop_size in patch_sizes:
        multi_scale_crops.append(
            A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    A.RandomCrop(height=crop_size, width=crop_size),
                    A.Resize(height=patch_size, width=patch_size),
                ]
            )
        )

    return A.Compose(
        [
            A.OneOf(multi_scale_crops, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.25, 0.25),
                    A.HueSaturationValue(12, 25, 25),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.04, 0.24)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                    A.MedianBlur(blur_limit=3),
                ],
                p=0.35,
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.20),
            # NOTE: A.Affine with scale was removed — the multi-scale crop block
            # above already handles scale augmentation, and double-scaling mis-aligns
            # the segmentation mask relative to the image.
            # CoarseDropout: simulates tree canopy occlusion
            A.CoarseDropout(
                num_holes_range=(8, 16),
                hole_height_range=(32, 64),
                hole_width_range=(32, 64),
                fill=0,
                p=0.35,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(patch_size: int = 640):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=patch_size,
                min_width=patch_size,
                border_mode=cv2.BORDER_REFLECT,
            ),
            A.CenterCrop(height=patch_size, width=patch_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_clf_train_transforms(
    crop_size: int = 160,
    use_randaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 7,
):
    transforms = [
        # scale=(0.80,1.0): lower 0.70 was too aggressive — produced 157px
        # texture snippets that barely showed a full rooftop at 224px crop size.
        A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.80, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.Transpose(p=0.5),
    ]
    if use_randaugment:
        transforms.append(_randaugment_block(randaugment_n, randaugment_m, p=0.85))
    transforms.extend(
        [
            A.ColorJitter(0.35, 0.35, 0.25, 0.08, p=0.7),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.16)),  # replaces var_limit
                    A.GaussianBlur(blur_limit=3),
                    A.Sharpen(alpha=(0.2, 0.5)),
                ],
                p=0.35,
            ),
            # Simulate shadow (rooftop lighting variation)
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.10),
            A.RandomSunFlare(
                src_radius=80,
                num_flare_circles_range=(2, 6),
                p=0.15,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(
        transforms
    )


def _randaugment_block(n: int = 2, m: int = 7, p: float = 0.85):
    """Albumentations RandAugment when available, otherwise a close fallback."""

    if hasattr(A, "RandAugment"):
        return A.RandAugment(num_ops=int(n), magnitude=int(m), p=p)

    magnitude = max(1, min(int(m), 10))
    rotate = 3 * magnitude
    color = 0.03 * magnitude
    ops = [
        A.RandomBrightnessContrast(color, color, p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=2 * magnitude,
            sat_shift_limit=4 * magnitude,
            val_shift_limit=3 * magnitude,
            p=1.0,
        ),
        A.Sharpen(alpha=(0.05, min(0.8, 0.08 * magnitude)), p=1.0),
        A.Affine(
            rotate=(-rotate, rotate),
            translate_percent=(-0.01 * magnitude, 0.01 * magnitude),
            scale=(1.0 - 0.02 * magnitude, 1.0 + 0.02 * magnitude),
            border_mode=cv2.BORDER_REFLECT,
            p=1.0,
        ),
    ]
    # Posterize and Equalize API changed across albumentations versions—guard both
    if hasattr(A, "Posterize"):
        try:
            ops.append(A.Posterize(num_bits=max(4, 8 - magnitude // 2), p=1.0))
        except TypeError:
            pass  # older API may not accept num_bits kwarg
    if hasattr(A, "Equalize"):
        ops.append(A.Equalize(p=1.0))
    if hasattr(A, "SomeOf"):
        return A.SomeOf(ops, n=int(n), replace=False, p=p)
    return A.OneOf(ops, p=p)


def get_clf_val_transforms(crop_size: int = 160):
    return A.Compose(
        [
            A.Resize(height=crop_size, width=crop_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL SPLIT  (in-memory, no symlinks, no file copies)
# ─────────────────────────────────────────────────────────────────────────────


def split_clf_dataset(
    root_dir,
    class_names,
    val_fraction=0.15,
    seed=42,
    crop_size=160,
    use_randaugment=False,
    randaugment_n=2,
    randaugment_m=7,
):
    c2id = {c: i for i, c in enumerate(class_names)}
    trn_samples = []
    val_samples = []
    rng = random.Random(seed)

    for cls in class_names:
        cls_dir = Path(root_dir) / cls
        if not cls_dir.is_dir():
            continue
        cls_samples = [(p, c2id[cls]) for p in cls_dir.glob("*.png") if p.stat().st_size > 0]
        if not cls_samples:
            continue
        cls_samples.sort(key=lambda x: str(x[0]))
        rng.shuffle(cls_samples)

        n_val = max(1, int(len(cls_samples) * val_fraction)) if len(cls_samples) > 1 else 0
        val_samples.extend(cls_samples[:n_val])
        trn_samples.extend(cls_samples[n_val:])

    if not trn_samples:
        raise ValueError(f"No valid training samples found in {root_dir}")
    if not val_samples:
        val_samples = trn_samples[:max(1, len(trn_samples) // 10)] if len(trn_samples) > 10 else []
        trn_samples = trn_samples[len(val_samples):]

    rng.shuffle(trn_samples)
    rng.shuffle(val_samples)

    train_ds = RooftopDataset(
        root_dir,
        class_names,
        crop_size,
        is_train=True,
        samples=trn_samples,
        use_randaugment=use_randaugment,
        randaugment_n=randaugment_n,
        randaugment_m=randaugment_m,
    )
    val_ds = RooftopDataset(
        root_dir, class_names, crop_size, is_train=False, samples=val_samples
    )
    return train_ds, val_ds


def split_dataset(
    img_dir,
    mask_dir,
    val_fraction=0.15,
    seed=42,
    num_classes=4,
    patch_size=640,
    patch_sizes=None,
):
    mask_dir_p = Path(mask_dir)
    all_imgs = [p for p in Path(img_dir).glob("*.png") if p.stat().st_size > 0]
    if not all_imgs:
        raise ValueError(f"No valid PNG patches found in {img_dir}")

    rng = random.Random(seed)

    def _fg_ratio(img_path: Path) -> float:
        mp = mask_dir_p / img_path.name
        if not mp.exists() or mp.stat().st_size == 0:
            return 0.0
        try:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.size == 0:
                return 0.0
            return float((m > 0).mean())
        except Exception:
            return 0.0

    ratios = [_fg_ratio(p) for p in all_imgs]
    sorted_idx = sorted(range(len(all_imgs)), key=lambda i: ratios[i])
    n = len(sorted_idx)
    strata_size = max(1, n // 4)
    strata = [sorted_idx[i : i + strata_size] for i in range(0, n, strata_size)]

    trn_imgs, val_imgs = [], []
    for stratum in strata:
        rng.shuffle(stratum)
        n_val = max(1, int(len(stratum) * val_fraction)) if len(stratum) > 1 else 0
        val_imgs.extend(all_imgs[i] for i in stratum[:n_val])
        trn_imgs.extend(all_imgs[i] for i in stratum[n_val:])

    if not trn_imgs:
        trn_imgs = all_imgs
        val_imgs = []
    elif not val_imgs:
        val_imgs = trn_imgs[:max(1, len(trn_imgs) // 10)] if len(trn_imgs) > 10 else []
        trn_imgs = trn_imgs[len(val_imgs):]

    rng.shuffle(trn_imgs)
    rng.shuffle(val_imgs)

    train_ds = SegmentationDataset(
        img_dir, mask_dir, patch_size, patch_sizes,
        num_classes, is_train=True, file_list=trn_imgs,
    )
    val_ds = SegmentationDataset(
        img_dir, mask_dir, patch_size, patch_sizes,
        num_classes, is_train=False, file_list=val_imgs,
    )
    return train_ds, val_ds
