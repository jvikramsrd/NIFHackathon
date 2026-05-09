"""
models/stage2_models.py  (v3 — Accuracy-Maximised)
────────────────────────────────────────────────────
Stage 2A improvements:
  • ArcFace loss head for tighter inter-class angular margins
  • 3-scale TTA (0.875×, 1.0×, 1.25×) with 8-fold D4 per scale
  • Per-crop instance normalization before augmentation
  • drop_path_rate from config

Stage 2B improvements:
  • SAHI (Slicing Aided Hyper Inference) for small object recall
  • Per-class confidence thresholds (transformer high / well low)
  • Soft-NMS Gaussian (unchanged, verified correct)
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace angular-margin classification head
# ─────────────────────────────────────────────────────────────────────────────


class ArcFaceHead(nn.Module):
    """
    ArcFace (Deng et al., 2019) additive angular margin loss head.

    Pushes class embeddings further apart in angular (cosine) space,
    reducing confusion between visually-similar categories (RCC vs Tiled).

    During inference (labels=None) returns cosine logits scaled by s.
    During training (labels provided) returns angular-margin logits for loss.
    """

    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # Pre-compute cos(m) and sin(m) for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if labels is None:
            # Inference: return scaled cosine logits
            return cosine * self.s
        # Training: add angular margin to the target class
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A  —  ConvNeXt-Large Rooftop Classifier with ArcFace
# ─────────────────────────────────────────────────────────────────────────────


class RooftopClassifier(nn.Module):
    """
    ConvNeXt-Large fine-tuned for 4-class rooftop classification with:
    - ArcFace angular-margin head for tighter class separation
    - 3-scale × 8-fold (or 4-fold) TTA at inference
    - MixUp + CutMix training augmentation
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]
        self.use_arcface = cfg.get("use_arcface", True)

        self.backbone = timm.create_model(
            cfg["arch"],
            pretrained=cfg["pretrained"],
            num_classes=0,
            global_pool="avg",
            drop_path_rate=cfg.get("drop_path_rate", 0.4),
        )
        in_features = self.backbone.num_features  # 1536 for convnext_large

        # Scale hidden dim proportionally to backbone features
        hidden_dim = max(512, in_features // 2)

        # Feature projection trunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.50),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.30),
        )

        if self.use_arcface:
            self.head = ArcFaceHead(hidden_dim, self.num_classes, s=30.0, m=0.50)
        else:
            self.head = nn.Linear(hidden_dim, self.num_classes)

        # Cross-entropy with reduced label smoothing (0.05 vs old 0.10)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.05)
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.trunk(self.backbone(x))
        if self.use_arcface:
            return self.head(feats, labels)
        return self.head(feats)

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    def forward_train(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass that passes labels to ArcFace head for margin computation."""
        feats = self.trunk(self.backbone(x))
        if self.use_arcface:
            return self.head(feats, labels)
        return self.head(feats)

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

    # ── 3-Scale × D4 TTA ───────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, tta_steps: int = 16, return_probs: bool = False
    ) -> torch.Tensor:
        """
        3-scale TTA with D4 symmetry group:
        Scales: 0.875× (wider context), 1.0× (base), 1.25× (fine texture)
        Per scale: 8 folds (4 rotations × 2 flip states)

        Scale weights: 0.875×→0.8, 1.0×→1.0, 1.25×→0.9 (base is authoritative).
        Total weighted passes: up to 24 (3 scales × 8 folds).
        """
        self.eval()
        _, _, H, W = x.shape
        all_probs = []
        scale_weights = []

        scales_config = [
            (0.875, 0.8),   # zoom out — captures larger buildings wholly
            (1.0,   1.0),   # base scale
            (1.25,  0.9),   # zoom in — captures fine texture (RCC grit, tile ridges)
        ]

        n_folds = min(tta_steps, 8)  # up to 8 folds per scale

        for scale, weight in scales_config:
            if scale != 1.0:
                crop_H = int(H * (1.0 / scale))
                crop_W = int(W * (1.0 / scale))
                if scale < 1.0:
                    # zoom out: pad then center-crop (shows more context)
                    pad_h = max(0, (crop_H - H) // 2)
                    pad_w = max(0, (crop_W - W) // 2)
                    x_s = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
                    x_s = F.interpolate(x_s, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    # zoom in: center-crop then resize up
                    sh = int(H / scale)
                    sw = int(W / scale)
                    sy = (H - sh) // 2
                    sx = (W - sw) // 2
                    x_s = x[:, :, sy: sy + sh, sx: sx + sw]
                    x_s = F.interpolate(x_s, size=(H, W), mode="bilinear", align_corners=False)
            else:
                x_s = x

            for k in range(n_folds):
                aug = torch.rot90(x_s, k % 4, dims=[2, 3])
                if k >= 4:
                    aug = torch.flip(aug, [3])
                # ArcFace: labels=None → returns cosine logits
                logit = self(aug)
                prob = torch.softmax(logit, 1)
                all_probs.append(prob)
                scale_weights.append(weight)

        weights_t = torch.tensor(scale_weights, device=x.device, dtype=torch.float32)  # (N_aug,)
        stacked = torch.stack(all_probs, dim=0)  # (N_aug, B, C)
        # Weighted sum then normalise
        mean_probs = (stacked * weights_t.view(-1, 1, 1)).sum(0) / weights_t.sum()

        if return_probs:
            return mean_probs
        return mean_probs.argmax(1)

    def class_weights(self) -> torch.Tensor:
        """Return uniform weights (override if you have counts)."""
        return torch.ones(self.num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B  —  YOLOv9 Infrastructure Detector with SAHI + per-class thresholds
# ─────────────────────────────────────────────────────────────────────────────


class InfrastructureDetector:
    """
    YOLOv9 detector with:
    - SAHI sliced inference for small object recall (transformers, wells)
    - Per-class confidence thresholds (transformer high / well low)
    - Oriented bounding-box (OBB) support
    """

    def __init__(self, cfg: dict, ckpt_dir: str = "checkpoints"):
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self._backend = None
        self.model: Any = None
        self._sahi_model: Any = None
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO  # type: ignore

            variant = self.cfg.get("model_variant", "yolov9e")
            if self.cfg.get("use_obb"):
                variant = self.cfg.get("obb_model_variant", f"{variant}-obb")
            try:
                self.model = YOLO(f"{variant}.pt")
            except Exception as exc:
                if not self.cfg.get("use_obb"):
                    raise
                fallback = self.cfg.get("model_variant", "yolov9e")
                log.warning(
                    "Could not load OBB model %s.pt (%s); falling back to %s.pt",
                    variant, exc, fallback,
                )
                self.model = YOLO(f"{fallback}.pt")
                self.cfg["use_obb"] = False
                variant = fallback
            self._backend = "yolo"
            log.info(f"  YOLO loaded: {variant}")
        except ImportError:
            log.warning("ultralytics not found → falling back to Faster R-CNN")
            self._backend = "frcnn"
            self.model = _build_frcnn(self.cfg["num_classes"])

    def train(self, data_yaml: str, device: str = "0"):
        if self._backend != "yolo":
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        train_args = dict(
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
            copy_paste=self.cfg.get("copy_paste", 0.30),
            cache=self.cfg.get("cache", "ram"),
            device=device,
            project=str(self.ckpt_dir),
            name=f"stage2b_{self.cfg['model_variant']}",
            exist_ok=True,
            amp=True,
            verbose=True,
            workers=self.cfg.get("workers", 0),
        )
        if self.cfg.get("use_obb"):
            train_args["task"] = "obb"
        return self.model.train(**train_args)

    def _get_sahi_model(self):
        """Lazily build the SAHI AutoDetectionModel wrapper."""
        if self._sahi_model is not None:
            return self._sahi_model
        try:
            from sahi import AutoDetectionModel  # type: ignore
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model=self.model,
                confidence_threshold=self.cfg.get("conf_thresh", 0.10),
                device="cuda:0",
            )
            return self._sahi_model
        except ImportError:
            log.warning("sahi not installed; falling back to standard inference. "
                        "Install with: pip install sahi")
            return None

    def predict(self, img_path: str) -> list:
        if self._backend != "yolo":
            return []

        use_sahi = self.cfg.get("use_sahi", True)
        class_thresholds: Dict[str, float] = self.cfg.get(
            "class_conf_thresh",
            {"transformer": 0.20, "overhead_tank": 0.12, "well": 0.08},
        )
        default_thresh = self.cfg.get("conf_thresh", 0.10)

        raw_dets: list = []

        if use_sahi:
            sahi_model = self._get_sahi_model()
            if sahi_model is not None:
                try:
                    from sahi.predict import get_sliced_prediction  # type: ignore
                    result = get_sliced_prediction(
                        img_path,
                        sahi_model,
                        slice_height=640,
                        slice_width=640,
                        overlap_height_ratio=0.30,
                        overlap_width_ratio=0.30,
                        perform_standard_pred=True,
                        postprocess_type="NMM",  # Non-Maximum Merging
                        postprocess_match_threshold=0.50,
                        verbose=0,
                    )
                    for pred in result.object_prediction_list:
                        bbox = pred.bbox
                        # Use category name for robust class lookup (not raw ID)
                        pred_cls_name = pred.category.name.lower()
                        # Find matching class index in our config
                        cid = next(
                            (i for i, c in enumerate(self.cfg["class_names"]) if c.lower() == pred_cls_name),
                            pred.category.id,
                        )
                        raw_dets.append({
                            "class_id": cid,
                            "class_name": self.cfg["class_names"][cid] if cid < len(self.cfg["class_names"]) else pred_cls_name,
                            "bbox_xyxy": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                            "obb_xywhr": None,
                            "conf": float(pred.score.value),
                        })
                except Exception as e:
                    log.warning("SAHI inference failed (%s); falling back to standard.", e)
                    use_sahi = False

        if not use_sahi or not raw_dets:
            results = self.model(
                img_path,
                conf=min(class_thresholds.values()) if class_thresholds else default_thresh,
                iou=self.cfg["iou_thresh"],
                max_det=self.cfg.get("max_det", 300),
                augment=True,
            )
            for r in results:
                obb = getattr(r, "obb", None) if self.cfg.get("use_obb") else None
                boxes = obb if obb is not None and getattr(obb, "xyxy", None) is not None else r.boxes
                xyxy_values = boxes.xyxy
                for i, box in enumerate(boxes):
                    cid = int(box.cls)
                    xywhr_tensor = getattr(box, "xywhr", None)
                    xywhr_list = xywhr_tensor.squeeze(0).tolist() if xywhr_tensor is not None else None
                    raw_dets.append({
                        "class_id": cid,
                        "class_name": self.cfg["class_names"][cid],
                        "bbox_xyxy": xyxy_values[i].tolist(),
                        "obb_xywhr": xywhr_list,
                        "conf": float(box.conf),
                    })

        # Apply per-class confidence thresholds
        out = []
        for det in raw_dets:
            cls_name = det.get("class_name", "")
            thresh = class_thresholds.get(cls_name, default_thresh)
            if det["conf"] >= thresh:
                out.append(det)

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
        max_idx = scores.argmax()
        kept.append(indices[max_idx].item())
        kept_scores.append(scores[max_idx].item())
        max_box = boxes[max_idx]

        mask = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
        mask[max_idx] = False
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]

        if len(boxes) == 0:
            break

        x1 = torch.max(boxes[:, 0], max_box[0])
        y1 = torch.max(boxes[:, 1], max_box[1])
        x2 = torch.min(boxes[:, 2], max_box[2])
        y2 = torch.min(boxes[:, 3], max_box[3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_b = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        iou = inter / (area_a + area_b - inter + 1e-6)

        decay = torch.exp(-(iou ** 2) / sigma)
        scores *= decay

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
