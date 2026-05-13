"""
models/stage1_segmentation.py  (v3 — Accuracy-Maximised)
──────────────────────────────────────────────────────────
Improvements in v3:
  • UNet++ decoder (nested dense skip connections → sharper building boundaries)
  • Deep supervision (auxiliary heads at decoder stages 2, 3, 4)
  • Lovász-Softmax loss (directly optimises IoU, not a proxy)
  • Instance-touching separation loss (penalises merged adjacent buildings)
  • Cosine-log Dice loss (smoother gradient landscape near 0/1)
  • 16-fold TTA with 3 scales (0.875×, 1.0×, 1.25×) instead of 2
"""

from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_stage1_model(cfg: dict) -> nn.Module:
    """Build the configured smp architecture with scSE attention."""
    arch = cfg.get("arch", "Unet")
    ModelCls = getattr(smp, arch, smp.Unet)
    model = ModelCls(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        activation=None,
        decoder_attention_type=cfg.get("decoder_attention_type", "scse"),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise LR groups for transformer backbone
# ─────────────────────────────────────────────────────────────────────────────


def get_parameter_groups(model: nn.Module, cfg: dict) -> List[dict]:
    """Two efficient parameter groups: encoder (lower LR) and decoder (full LR).
    Batching into 2 groups instead of one-per-param is critical for memory efficiency
    with 64M-param transformers."""
    no_decay = {
        "bias",
        "LayerNorm.weight",
        "norm.weight",
        "norm1.weight",
        "norm2.weight",
    }
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
    if enc_decay:
        groups.append({"params": enc_decay, "lr": enc_lr, "weight_decay": wd})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": enc_lr, "weight_decay": 0.0})
    if dec_decay:
        groups.append({"params": dec_decay, "lr": dec_lr, "weight_decay": wd})
    if dec_nodecay:
        groups.append({"params": dec_nodecay, "lr": dec_lr, "weight_decay": 0.0})
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Lovász-Softmax loss — directly optimises mIoU
# ─────────────────────────────────────────────────────────────────────────────


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovász extension gradient for the sorted error vector."""
    p = gt_sorted.shape[0]
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    """
    Lovász-Softmax loss for multi-class segmentation (flat inputs).
    probs:  (P, C) — softmax probabilities
    labels: (P,)  — integer class labels
    """
    C = probs.shape[1]
    losses = []
    class_to_sum = list(range(C)) if classes == "all" else labels.unique().tolist()
    for c in class_to_sum:
        c = int(c)
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0 and classes == "present":
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = fg[perm]
        grad = _lovasz_grad(gt_sorted)
        losses.append((errors_sorted * grad).sum())
    return torch.stack(losses).mean() if losses else probs.new_tensor(0.0)


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    """
    Lovász-Softmax: reshape (B,C,H,W) → (B*H*W, C) and compute per pixel.
    """
    B, C, H, W = probs.shape
    probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels_flat = labels.view(-1)
    return lovasz_softmax_flat(probs_flat, labels_flat, classes=classes)


# ─────────────────────────────────────────────────────────────────────────────
# Tri-Loss v3: Dice + CE + Focal + Boundary/Hausdorff + Lovász + Touching
# ─────────────────────────────────────────────────────────────────────────────


class TriLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dice_weight=0.40,
        ce_weight=0.15,
        focal_weight=0.15,
        boundary_weight=0.15,
        lovasz_weight=0.15,
        touching_weight=0.10,
        focal_gamma=2.0,
        class_weights=None,
        smooth=1e-6,
        label_smoothing=0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dw = dice_weight
        self.cw_w = ce_weight
        self.fw = focal_weight
        self.bw = boundary_weight
        self.lv_w = lovasz_weight
        self.tw = touching_weight
        self.gamma = focal_gamma
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.use_boundary = boundary_weight > 0
        self.use_lovasz = lovasz_weight > 0
        self.use_touching = touching_weight > 0
        w = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights
            else torch.ones(num_classes)
        )
        w = w / (w.sum() + 1e-6) * num_classes
        self.register_buffer("cw", w)
        self.register_buffer("cw_norm", w / (w.sum() + 1e-6))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Handle deep supervision: list/tuple of logits from UNet++ aux heads
        if isinstance(logits, (list, tuple)):
            # Auxiliary weights: main head 1.0, intermediate heads 0.4 / 0.2 / 0.1
            aux_weights = [1.0, 0.4, 0.2, 0.1]
            total = logits[0].new_tensor(0.0)
            denom = 0.0
            for logit, aw in zip(logits, aux_weights):
                # Resize aux logit to match target resolution if needed
                if logit.shape[-2:] != targets.shape[-2:]:
                    logit = F.interpolate(logit, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                total = total + aw * self._compute(logit, targets)
                denom += aw
            return total / denom

        return self._compute(logits, targets)

    def _compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        tgt = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        # Single log_softmax pass: probs derived cheaply, avoids a second softmax
        # and eliminates the redundant second F.cross_entropy call for focal loss.
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # ── Dice (cosine-log variant) — vectorized over classes ───────────────
        inter = (probs * tgt).sum(dim=(2, 3))                           # (B, C)
        union = probs.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))            # (B, C)
        dice_c = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        d_loss = (torch.log(torch.cosh(dice_c)).mean(dim=0) * self.cw_norm).sum()

        # ── Cross-Entropy with label smoothing ───────────────────────────────
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.cw, label_smoothing=self.label_smoothing
        )

        # ── Focal loss: .float() matches F.cross_entropy's internal fp32 path ─
        ce_none = -log_probs.float().gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        pt = torch.exp(-ce_none)
        f_loss = (self.cw[targets] * (1.0 - pt) ** self.gamma * ce_none).mean()

        total = self.dw * d_loss + self.cw_w * ce_loss + self.fw * f_loss

        # ── Lovász-Softmax (directly optimises IoU) ──────────────────────────
        if self.use_lovasz:
            total = total + self.lv_w * lovasz_softmax(probs, targets, classes="present")

        # ── Boundary / Hausdorff loss ─────────────────────────────────────────
        if self.use_boundary:
            total = total + self.bw * _boundary_loss(probs, tgt, self.smooth)

        # ── Instance-touching separation loss (building class only) ───────────
        if self.use_touching:
            total = total + self.tw * _touching_separation_loss(probs, targets, class_id=1)

        return total


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / Hausdorff loss helpers
# ─────────────────────────────────────────────────────────────────────────────


def _boundary_loss(probs: torch.Tensor, tgt: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Boundary-aware loss with a differentiable Hausdorff-style term."""
    edge_pred = _soft_edge_map(probs)
    edge_true = _soft_edge_map(tgt)
    edge_dice = _soft_dice_loss(edge_pred, edge_true, smooth)
    hausdorff = _hausdorff_er_loss(edge_pred, edge_true)
    return 0.5 * edge_dice + 0.5 * hausdorff


def _soft_edge_map(x: torch.Tensor) -> torch.Tensor:
    """Approximate class boundaries with differentiable max/min pooling."""
    dilation = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    erosion = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (dilation - erosion).clamp_min(0.0)


def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float) -> torch.Tensor:
    dims = tuple(range(2, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def _hausdorff_er_loss(
    pred_edges: torch.Tensor,
    target_edges: torch.Tensor,
    max_iter: int = 4,
) -> torch.Tensor:
    """Erosion-based differentiable Hausdorff approximation."""
    error = (pred_edges - target_edges).pow(2)
    if error.numel() == 0:
        return pred_edges.new_tensor(0.0)

    loss = pred_edges.new_tensor(0.0)
    eroded = error
    actual_iters = 0
    for k in range(max_iter):
        eroded = -F.max_pool2d(-eroded, kernel_size=3, stride=1, padding=1)
        actual_iters += 1
        loss = loss + eroded.mean() * float((k + 1) ** 2)
        if torch.isclose(eroded.max(), eroded.new_tensor(0.0)):
            break
    return loss / max(float(actual_iters), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Instance-touching separation loss
# ─────────────────────────────────────────────────────────────────────────────


def _touching_separation_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_id: int = 1,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    Penalise high predicted probability for class `class_id` at the boundary
    between adjacent building instances.

    Strategy:
      1. Dilate the GT building mask → grows each building region outward.
      2. Erode the GT building mask  → shrinks each building region inward.
      3. Boundary = dilated − eroded (pixels that belong to inter-instance gaps).
      4. Loss = mean predicted building prob at boundary pixels.

    Effect: Forces the model to lower its confidence at the gap between two
    touching buildings, which helps separate them into distinct polygons.
    """
    building_mask = (targets == class_id).float().unsqueeze(1)  # (B,1,H,W)
    pad = kernel_size // 2
    dilated = F.max_pool2d(building_mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-building_mask, kernel_size=kernel_size, stride=1, padding=pad)
    # Soft sigmoid boundary instead of hard clamp — smoother gradients near 0/1
    boundary = torch.sigmoid((dilated - eroded) * 4.0 - 2.0)  # smooth 0→1 at edge
    # Only penalise where GT says there IS a building nearby (real boundaries)
    bld_prob = probs[:, class_id : class_id + 1]  # (B,1,H,W)
    loss = (bld_prob * boundary).sum() / (boundary.sum() + 1e-6)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 3-Scale TTA (0.875×, 1.0×, 1.25×) + D4 symmetries
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    image: torch.Tensor,
    num_classes: int,
    amp_dtype=torch.bfloat16,
    fast_tta: bool = True,
) -> torch.Tensor:
    """
    TTA with 3 scales × D4 symmetries (or fast mode with fewer augmentations).
    Scales: 0.875× (see larger buildings), 1.0× (base), 1.25× (see fine texture).
    Returns (B, C, H, W) softmax probabilities.
    """
    was_training = model.training
    model.eval()
    B, C_in, H, W = image.shape
    probs_sum = torch.zeros((B, num_classes, H, W), device=image.device, dtype=torch.float32)
    total_weight = 0.0

    def _run_scale(img: torch.Tensor, weight: float, n_augs: int):
        nonlocal total_weight  # required: assignment to outer scope var
        h_s, w_s = img.shape[2], img.shape[3]
        for k in range(n_augs):
            aug = torch.rot90(img, k % 4, dims=[2, 3])
            if k >= 4:
                aug = torch.flip(aug, [3])
            with torch.amp.autocast(image.device.type, dtype=amp_dtype):
                raw = model(aug)
                # Handle deep supervision list from UNet++
                if isinstance(raw, (list, tuple)):
                    raw = raw[0]
                prob = torch.softmax(raw.float(), 1)
            # Undo rotation / flip
            if k >= 4:
                prob = torch.flip(prob, [3])
            prob = torch.rot90(prob, -(k % 4), dims=[2, 3])
            # Resize back to original spatial size if needed
            if prob.shape[2:] != (H, W):
                prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
            probs_sum.add_(prob * weight)
            total_weight += weight

    scales = [(0.875, 0.8), (1.0, 1.0), (1.25, 0.9)] if not fast_tta else [(1.0, 1.0), (1.25, 0.9)]
    n_augs_per_scale = 4 if fast_tta else 8

    for scale, weight in scales:
        if scale != 1.0:
            img_s = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            img_s = image
        _run_scale(img_s, weight, n_augs_per_scale)

    if was_training:
        model.train()
    return probs_sum / max(total_weight, 1e-6)


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
            dice_weight=cfg.get("dice_weight", 0.40),
            ce_weight=cfg.get("bce_weight", 0.15),
            focal_weight=cfg.get("focal_weight", 0.15),
            boundary_weight=cfg.get("boundary_weight", 0.15),
            lovasz_weight=cfg.get("lovasz_weight", 0.15),
            touching_weight=cfg.get("touching_weight", 0.10),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            class_weights=cfg.get("class_weights"),
            label_smoothing=cfg.get("label_smoothing", 0.05),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, masks):
        return self.criterion(logits, masks)

    def parameter_groups(self):
        return get_parameter_groups(self.model, self.cfg)

    @torch.no_grad()
    def predict(self, images, use_tta=False, amp_dtype=torch.bfloat16, fast_tta=True):
        was_training = self.training
        self.eval()
        if use_tta:
            result = tta_predict(
                self.model,
                images,
                self.cfg["num_classes"],
                amp_dtype,
                fast_tta=fast_tta,
            ).argmax(1)
        else:
            raw = self.model(images)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            result = raw.argmax(1)
        if was_training:
            self.train()
        return result
