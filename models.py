import math
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import timm

from utils.core import get_logger

log = get_logger(__name__)

# ==============================================================================
# STAGE 1: SEGMENTATION MODEL
# ==============================================================================

class TriLoss(nn.Module):
    """Simplified TriLoss: Focal + Lovasz + Dice"""
    def __init__(self, num_classes, focal_weight=0.25, lovasz_weight=0.40, dice_weight=0.15, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.fw = focal_weight
        self.lv_w = lovasz_weight
        self.dw = dice_weight
        
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else torch.ones(num_classes)
        w = w / (w.sum() + 1e-6) * num_classes
        self.register_buffer("cw", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or targets.numel() == 0:
            return logits.new_tensor(0.0)
            
        C = logits.shape[1]
        targets = torch.clamp(targets, 0, C - 1)
        tgt = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        
        probs = torch.softmax(logits, dim=1)
        ce_none = F.cross_entropy(logits.float(), targets, reduction="none")
        pt = torch.exp(-ce_none)
        
        f_loss = (self.cw[targets] * (1.0 - pt) ** 3.0 * ce_none).mean()
        total = self.fw * f_loss
        
        # Simplified Dice
        if self.dw > 0:
            dims = (0, 2, 3)
            inter = (probs * tgt).sum(dim=dims)
            denom = probs.sum(dim=dims) + tgt.sum(dim=dims)
            dice = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
            total += self.dw * dice
            
        return total

class Stage1Module(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        arch = cfg.get("arch", "Unet")
        ModelCls = getattr(smp, arch, smp.Unet)
        self.model = ModelCls(
            encoder_name=cfg["encoder"],
            encoder_weights=cfg["encoder_weights"],
            in_channels=cfg["in_channels"],
            classes=cfg["num_classes"],
            activation=None,
        )
        self.criterion = TriLoss(
            cfg["num_classes"], 
            class_weights=cfg.get("class_weights")
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, masks):
        return self.criterion(logits, masks)

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        raw = self.model(images)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        result = raw.argmax(1)
        return result


# ==============================================================================
# STAGE 2A: ROOFTOP CLASSIFIER
# ==============================================================================

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2: return x
        x = x.clamp(min=self.eps)
        return x.pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

class RooftopClassifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]

        self.backbone = timm.create_model(
            cfg["arch"],
            pretrained=cfg["pretrained"],
            num_classes=0,
            global_pool="",
        )
        self.gem = GeMPooling()
        in_features = self.backbone.num_features
        hidden_dim = max(512, in_features // 2)

        self.trunk = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.50),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, self.num_classes)
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        spatial = self.backbone(x)
        pooled = self.gem(spatial)
        return self.trunk(pooled)

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x).argmax(1)


# ==============================================================================
# STAGE 2B: INFRASTRUCTURE DETECTOR
# ==============================================================================

class InfrastructureDetector:
    def __init__(self, cfg: dict, ckpt_dir: str = "checkpoints"):
        self.cfg = cfg
        from ultralytics import YOLO
        variant = self.cfg.get("model_variant", "yolo11l")
        self.model = YOLO(f"{variant}.pt")

    def train(self, data_yaml: str):
        return self.model.train(
            data=data_yaml,
            epochs=self.cfg["epochs"],
            imgsz=self.cfg["img_size"],
            batch=self.cfg["batch_size"],
            project="checkpoints",
            name=f"stage2b_{self.cfg['model_variant']}"
        )

    def predict(self, img_source) -> list:
        results = self.model(img_source, conf=0.10, max_det=300)
        out = []
        for r in results:
            for i, box in enumerate(r.boxes):
                cid = int(box.cls)
                out.append({
                    "class_id": cid,
                    "class_name": self.cfg["class_names"][cid] if cid < len(self.cfg["class_names"]) else "unknown",
                    "bbox_xyxy": box.xyxy[i].tolist(),
                    "conf": float(box.conf)
                })
        return out
