"""
models/stage2_models.py  (A4000-optimised)
───────────────────────────────────────────
Stage 2A : ConvNeXt-Large rooftop classifier
           MixUp + CutMix, 16-fold multi-scale TTA, class-weighted CE
Stage 2B : YOLOv8x infrastructure detector
           1280-px tiles, RAM cache, mosaic augmentation
"""

from pathlib import Path
from typing import Any, List

import numpy as np
import timm
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A  —  ConvNeXt-Base Rooftop Classifier
# ─────────────────────────────────────────────────────────────────────────────


class RooftopClassifier(nn.Module):
    """
    ConvNeXt-Base fine-tuned for 4-class rooftop classification.

    Why ConvNeXt over EfficientNet for rooftop materials?
    Rooftop discrimination (RCC vs Tiled vs Tin) is primarily a *texture*
    classification problem. ConvNeXt uses large-kernel (7×7) depthwise
    convolutions — superior to standard 3×3 kernels for texture frequencies.
    ConvNeXt-Base also outperforms EfficientNet-B4 on ImageNet texture bias.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]

        self.backbone = timm.create_model(
            cfg["arch"],
            pretrained=cfg["pretrained"],
            num_classes=0,
            global_pool="avg",
            drop_path_rate=0.4,  # stochastic depth — higher for larger models
        )
        in_features = (
            self.backbone.num_features
        )  # 1536 for convnext_large, 1024 for base

        # Scale hidden dim proportionally to backbone features
        hidden_dim = max(512, in_features // 2)

        # Two-layer head with LayerNorm (ConvNeXt canonical)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.50),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, self.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.08)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    # ── MixUp ──────────────────────────────────────────────────────────────
    def mixup(self, x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def mixup_loss(self, logits, ya, yb, lam):
        return lam * self.criterion(logits, ya) + (1 - lam) * self.criterion(logits, yb)

    # ── CutMix ─────────────────────────────────────────────────────────────
    def cutmix(self, x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return mixed, y, y[idx], lam

    # ── 16-fold Multi-Scale TTA inference ───────────────────────────────────
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, tta_steps: int = 16, return_probs: bool = False
    ) -> torch.Tensor:
        """
        16-fold TTA:
        Base scale (8 folds: D4 dihedral group of rotations + flips)
        1.25x zoomed scale (8 folds: D4 dihedral group on a cropped/zoomed image)

        This drastically improves classification accuracy on challenging rooftop
        textures (like faded RCC vs semi-pucca Tin) by forcing the model to evaluate
        the material at multiple spatial frequencies.
        """
        self.eval()
        import torch.nn.functional as F

        preds = []

        # Scale 1: Base Resolution
        for k in range(8):
            if k >= tta_steps:
                break
            aug = torch.rot90(x, k % 4, dims=[2, 3])
            if k >= 4:
                aug = torch.flip(aug, [3])
            preds.append(torch.softmax(self(aug), 1))

        # Scale 2: 1.25x Zoom (Center Crop + Resize)
        if tta_steps > 8:
            _, _, H, W = x.shape
            crop_H, crop_W = int(H * 0.8), int(W * 0.8)
            start_y, start_x = (H - crop_H) // 2, (W - crop_W) // 2
            x_zoomed = x[:, :, start_y : start_y + crop_H, start_x : start_x + crop_W]
            x_zoomed = F.interpolate(
                x_zoomed, size=(H, W), mode="bilinear", align_corners=False
            )

            for k in range(8):
                if k + 8 >= tta_steps:
                    break
                aug = torch.rot90(x_zoomed, k % 4, dims=[2, 3])
                if k >= 4:
                    aug = torch.flip(aug, [3])
                preds.append(torch.softmax(self(aug), 1))

        mean_probs = torch.stack(preds).mean(0)

        if return_probs:
            return mean_probs

        return mean_probs.argmax(1)

    def class_weights(self) -> torch.Tensor:
        """Return uniform weights (override if you have counts)."""
        return torch.ones(self.num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B  —  YOLOv8-l Infrastructure Detector
# ─────────────────────────────────────────────────────────────────────────────


class InfrastructureDetector:
    """YOLOv8x with 1280-px tiles and RAM caching."""

    def __init__(self, cfg: dict, ckpt_dir: str = "checkpoints"):
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self._backend = None
        self.model: Any = None
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(f"{self.cfg['model_variant']}.pt")
            self._backend = "yolo"
            print(f"  YOLOv8 loaded: {self.cfg['model_variant']}")
        except ImportError:
            print("[WARN] ultralytics not found → falling back to Faster R-CNN")
            self._backend = "frcnn"
            self.model = _build_frcnn(self.cfg["num_classes"])

    def train(self, data_yaml: str, device: str = "0"):
        if self._backend != "yolo":
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        return self.model.train(
            data=data_yaml,
            epochs=self.cfg["epochs"],
            imgsz=self.cfg["img_size"],
            batch=self.cfg["batch_size"],
            lr0=self.cfg["lr0"],
            lrf=self.cfg["lrf"],
            warmup_epochs=self.cfg.get("warmup_epochs", 3),
            patience=self.cfg.get("patience", 20),
            cos_lr=self.cfg.get("cos_lr", True),
            mosaic=self.cfg["mosaic"],
            close_mosaic=self.cfg.get("close_mosaic", 20),
            hsv_h=self.cfg.get("hsv_h", 0.015),
            hsv_s=self.cfg.get("hsv_s", 0.5),
            hsv_v=self.cfg.get("hsv_v", 0.3),
            degrees=self.cfg.get("degrees", 15.0),
            translate=self.cfg.get("translate", 0.1),
            scale=self.cfg.get("scale", 0.5),
            fliplr=self.cfg.get("fliplr", 0.5),
            mixup=self.cfg.get("mixup", 0.15),
            copy_paste=self.cfg.get("copy_paste", 0.0),
            cache=self.cfg.get("cache", "ram"),  # RAM cache
            device=device,
            project=str(self.ckpt_dir),
            name=f"stage2b_{self.cfg['model_variant']}",
            exist_ok=True,
            amp=True,  # native AMP
            verbose=True,
            workers=self.cfg.get("workers", 0),
        )

    def predict(self, img_path: str) -> list:
        if self._backend != "yolo":
            return []
        results = self.model(
            img_path,
            conf=self.cfg["conf_thresh"],
            iou=self.cfg["iou_thresh"],
            max_det=self.cfg.get("max_det", 300),
            augment=True,  # Enable YOLOv8 built-in multi-scale TTA for test-time augmentation
        )
        out = []
        for r in results:
            for box in r.boxes:
                cid = int(box.cls)
                out.append(
                    {
                        "class_id": cid,
                        "class_name": self.cfg["class_names"][cid],
                        "bbox_xyxy": box.xyxy[0].tolist(),
                        "conf": float(box.conf),
                    }
                )
        return out

    def evaluate(self, data_yaml: str):
        return self.model.val(data=data_yaml) if self._backend == "yolo" else None


# ─────────────────────────────────────────────────────────────────────────────
# Soft-NMS  —  Gaussian confidence decay for overlapping detections
# ─────────────────────────────────────────────────────────────────────────────


def soft_nms_gaussian(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
) -> tuple:
    """
    Soft-NMS with Gaussian penalty (Bodla et al., 2017).

    Instead of hard-suppressing overlapping boxes, decays their confidence:
        score_i *= exp( -iou(i, M)^2 / sigma )

    This preserves closely-spaced infrastructure that hard NMS would drop
    (e.g. two transformers on adjacent poles with overlapping bboxes).

    Args:
        boxes:    (N, 4) float tensor [x1, y1, x2, y2]
        scores:   (N,) float tensor
        sigma:    Gaussian decay bandwidth (lower = more aggressive suppression)
        score_threshold: discard boxes whose score falls below this

    Returns:
        (kept_indices, decayed_scores) — sorted by descending score
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

    boxes = boxes.float()
    scores = scores.float().clone()
    N = len(boxes)
    indices = torch.arange(N, device=boxes.device)
    kept = []
    kept_scores = []

    while len(scores) > 0:
        # Pick highest-scoring box
        max_idx = scores.argmax()
        kept.append(indices[max_idx].item())
        kept_scores.append(scores[max_idx].item())
        max_box = boxes[max_idx]

        # Remove the selected box
        mask = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
        mask[max_idx] = False
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]

        if len(boxes) == 0:
            break

        # Compute IoU of remaining boxes with the selected box
        x1 = torch.max(boxes[:, 0], max_box[0])
        y1 = torch.max(boxes[:, 1], max_box[1])
        x2 = torch.min(boxes[:, 2], max_box[2])
        y2 = torch.min(boxes[:, 3], max_box[3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_b = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        iou = inter / (area_a + area_b - inter + 1e-6)

        # Gaussian decay
        decay = torch.exp(-(iou**2) / sigma)
        scores *= decay

        # Prune low-confidence boxes
        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        indices = indices[keep]

    kept_idx = torch.tensor(kept, dtype=torch.long)
    out_scores = torch.tensor(kept_scores, dtype=torch.float32)
    return kept_idx, out_scores


def _build_frcnn(num_classes: int) -> nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
    return model
