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
        num_classes=4,
        transform=None,
        is_train=True,
        file_list=None,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes
        # Accept explicit file list (for train/val split without copying files)
        if file_list is not None:
            self.img_paths = [Path(p) for p in file_list]
        else:
            self.img_paths = sorted(self.img_dir.glob("*.png"))
        assert len(self.img_paths) > 0, f"No PNG patches in {img_dir}"
        self.transform = transform or (
            get_train_transforms(patch_size)
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
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
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
        crop_size=160,
        transform=None,
        is_train=True,
        samples=None,
    ):
        self.root = Path(root_dir)
        self.class_names = class_names
        self.c2id = {c: i for i, c in enumerate(class_names)}

        if samples is not None:
            self.samples = samples
        else:
            self.samples: List[Tuple[Path, int]] = []
            for cls in class_names:
                for p in (self.root / cls).glob("*.png"):
                    self.samples.append((p, self.c2id[cls]))
            self.samples.sort(key=lambda x: str(x[0]))

        assert len(self.samples) > 0, f"No crops in {root_dir}"

        self.transform = transform or (
            get_clf_train_transforms(crop_size)
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
        img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)

    def class_weights(self):
        counts = torch.zeros(len(self.class_names))
        for _, lbl in self.samples:
            counts[lbl] += 1
        w = 1.0 / (counts + 1e-6)
        return w / w.sum() * len(self.class_names)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# All tuned for drone aerial imagery (top-down perspective, no horizon)
# ─────────────────────────────────────────────────────────────────────────────


def get_train_transforms(patch_size: int = 640):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=patch_size,
                min_width=patch_size,
                border_mode=cv2.BORDER_REFLECT,
            ),
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
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
            # RandomFog: new API uses fog_coef_range tuple
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.15),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.04, 0.24)),  # replaces var_limit
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                    A.MedianBlur(blur_limit=3),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    
                    A.ElasticTransform(alpha=120, sigma=6),  # alpha_affine removed
                    
                    A.Perspective(scale=(0.05, 0.1)),
                    
                ],
                p=0.40,
            ),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45), p=0.5, border_mode=cv2.BORDER_REFLECT),
            # CoarseDropout: tuned heavily against tree occlusions
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


def get_clf_train_transforms(crop_size: int = 160):
    return A.Compose(
        [
            A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.70, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.Transpose(p=0.5),
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
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


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


def split_clf_dataset(root_dir, class_names, val_fraction=0.15, seed=42, crop_size=160):
    c2id = {c: i for i, c in enumerate(class_names)}
    trn_samples = []
    val_samples = []
    rng = random.Random(seed)

    for cls in class_names:
        cls_samples = [(p, c2id[cls]) for p in (Path(root_dir) / cls).glob("*.png")]
        cls_samples.sort(key=lambda x: str(x[0]))
        rng.shuffle(cls_samples)

        n_val = (
            max(1, int(len(cls_samples) * val_fraction)) if len(cls_samples) > 1 else 0
        )
        val_samples.extend(cls_samples[:n_val])
        trn_samples.extend(cls_samples[n_val:])

    rng.shuffle(trn_samples)
    rng.shuffle(val_samples)

    train_ds = RooftopDataset(
        root_dir, class_names, crop_size, is_train=True, samples=trn_samples
    )
    val_ds = RooftopDataset(
        root_dir, class_names, crop_size, is_train=False, samples=val_samples
    )
    return train_ds, val_ds


def split_dataset(
    img_dir, mask_dir, val_fraction=0.15, seed=42, num_classes=4, patch_size=640
):
    all_imgs = sorted(Path(img_dir).glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(all_imgs)

    n_val = max(1, int(len(all_imgs) * val_fraction))
    val_imgs = all_imgs[:n_val]
    trn_imgs = all_imgs[n_val:]

    # Pass explicit path lists directly — no symlinks, no copies
    train_ds = SegmentationDataset(
        img_dir, mask_dir, patch_size, num_classes, is_train=True, file_list=trn_imgs
    )
    val_ds = SegmentationDataset(
        img_dir, mask_dir, patch_size, num_classes, is_train=False, file_list=val_imgs
    )
    return train_ds, val_ds
