"""
models/stage1_segmentation.py  (A4000-optimised)
──────────────────────────────────────────────────
• UNet with Mix Transformer B4 encoder (mit_b4)
  - Natively supported by segmentation-models-pytorch
  - Hierarchical transformer — better than Swin for dense prediction
  - Similar VRAM usage to Swin-B at 640px patches
• Tri-loss: Dice + BCE + Focal
• 16-fold TTA with scale-jitter (4 rots × 2 flips × 2 scales)
• Layer-wise differential LR for transformer backbone
"""

from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_stage1_model(cfg: dict) -> nn.Module:
    model = smp.Unet(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        activation=None,
        decoder_attention_type="scse",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise LR groups for transformer backbone
# ─────────────────────────────────────────────────────────────────────────────


def get_parameter_groups(model: nn.Module, cfg: dict) -> List[dict]:
    """Two efficient parameter groups: encoder (lower LR) and decoder (full LR).
    Batching into 2 groups instead of one-per-param is critical for memory efficiency
    with 64M-param transformers."""
    no_decay = {"bias", "LayerNorm.weight", "norm.weight", "norm1.weight", "norm2.weight"}
    enc_decay, enc_nodecay, dec_decay, dec_nodecay = [], [], [], []
    enc_lr = cfg["lr"] * cfg["encoder_lr_mult"]
    dec_lr = cfg["lr"]
    wd = cfg["weight_decay"]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_enc = name.startswith("encoder")
        no_wd = any(nd in name for nd in no_decay)
        if is_enc:
            (enc_nodecay if no_wd else enc_decay).append(param)
        else:
            (dec_nodecay if no_wd else dec_decay).append(param)
    groups = []
    if enc_decay:   groups.append({"params": enc_decay,   "lr": enc_lr, "weight_decay": wd})
    if enc_nodecay: groups.append({"params": enc_nodecay, "lr": enc_lr, "weight_decay": 0.0})
    if dec_decay:   groups.append({"params": dec_decay,   "lr": dec_lr, "weight_decay": wd})
    if dec_nodecay: groups.append({"params": dec_nodecay, "lr": dec_lr, "weight_decay": 0.0})
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Tri-Loss: Dice + BCE + Focal
# ─────────────────────────────────────────────────────────────────────────────


class TriLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dice_weight=0.5,
        ce_weight=0.25,
        focal_weight=0.25,
        focal_gamma=2.0,
        class_weights=None,
        smooth=1e-6,
        label_smoothing=0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dw, self.cw_w, self.fw = dice_weight, ce_weight, focal_weight
        self.gamma = focal_gamma
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        w = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights
            else torch.ones(num_classes)
        )
        # Normalize class weights so Dice accumulation is properly scaled
        w = w / (w.sum() + 1e-6) * num_classes
        self.register_buffer("cw", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, C, H, W = logits.shape
        tgt = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)

        d_loss = logits.new_tensor(0.0)

        # Cross Entropy with label smoothing — prevents overconfident logits
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.cw, label_smoothing=self.label_smoothing
        )

        # Focal Loss
        ce_none = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_none)
        f_loss = (self.cw[targets] * (1 - pt) ** self.gamma * ce_none).mean()

        # Log-Cosh Dice — normalized so class weights sum correctly
        cw_sum = self.cw.sum() + 1e-6
        for c in range(C):
            p, t, w = probs[:, c], tgt[:, c], self.cw[c]
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_c = 1 - (2 * inter + self.smooth) / (union + self.smooth)
            dice_c = torch.log(torch.cosh(dice_c))
            d_loss = d_loss + (w / cw_sum) * dice_c.mean()

        return self.dw * d_loss + self.cw_w * ce_loss + self.fw * f_loss


# ─────────────────────────────────────────────────────────────────────────────
# 16-fold TTA
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def tta_predict(
    model: nn.Module, image: torch.Tensor, num_classes: int, amp_dtype=torch.bfloat16
) -> torch.Tensor:
    """8-fold TTA: D4 dihedral symmetries (4 rotations × 2 flips). Returns (C,H,W) softmax probs."""
    # Note: image is (1,C,H,W), output prob is (num_classes,H,W)
    model.eval()
    
    probs_sum = torch.zeros(
        (num_classes, image.shape[2], image.shape[3]), 
        device=image.device, 
        dtype=torch.float32
    )
    
    with torch.amp.autocast("cuda", dtype=amp_dtype):  # type: ignore
        for k in range(4):
            # Rotations
            aug = torch.rot90(image, k, dims=[2, 3])
            prob = torch.softmax(model(aug).float(), 1).squeeze(0)
            prob = torch.rot90(prob, -k, dims=[1, 2])
            probs_sum += prob
            
            # Flips + Rotations
            aug_f = torch.flip(image, [3])
            aug_f = torch.rot90(aug_f, k, dims=[2, 3])
            prob_f = torch.softmax(model(aug_f).float(), 1).squeeze(0)
            prob_f = torch.rot90(prob_f, -k, dims=[1, 2])
            prob_f = torch.flip(prob_f, [2])
            probs_sum += prob_f
            
    return probs_sum / 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Module wrapper
# ─────────────────────────────────────────────────────────────────────────────


class Stage1Module(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.model = build_stage1_model(cfg)
        self.criterion = TriLoss(
            cfg["num_classes"],
            cfg["dice_weight"],
            cfg["bce_weight"],  # interpreted as ce_weight
            cfg["focal_weight"],
            cfg["focal_gamma"],
            cfg["class_weights"],
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, masks):
        return self.criterion(logits, masks)

    def parameter_groups(self):
        return get_parameter_groups(self.model, self.cfg)

    @torch.no_grad()
    def predict(self, images, use_tta=False, amp_dtype=torch.bfloat16):
        self.eval()
        if use_tta:
            assert images.shape[0] == 1
            return (
                tta_predict(self.model, images, self.cfg["num_classes"], amp_dtype)
                .argmax(0)
                .unsqueeze(0)
            )
        return self.model(images).argmax(1)
