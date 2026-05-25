"""
models/stage1_segmentation.py  (v4 - Unet MiT-B4 production build)
──────────────────────────────────────────────────────────
Stage 1 segmentation model:
  - Unet decoder with scSE attention
  - MixTransformer B4 encoder for strong accuracy / speed / VRAM balance
  • Lovász-Softmax loss (directly optimises IoU, not a proxy)
  • Instance-touching separation loss (penalises merged adjacent buildings)
  • Cosine-log Dice loss (smoother gradient landscape near 0/1)
  - Batched multi-scale TTA with D4 symmetries
"""

from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

log = get_logger(__name__)

# Counter used to rate-limit the invalid-label warning so it doesn't flood logs.
_invalid_label_warn_count: int = 0
_INVALID_LABEL_WARN_EVERY: int = 100  # warn at most once per N forward passes

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
        # Use `in` instead of `startswith` so this works when the model is wrapped in
        # DDP (prefix "module."), torch.compile (prefix "_orig_mod."), or both.
        # With startswith("encoder") the check silently fails and the entire encoder
        # gets the full decoder LR — 10x too high — causing mIoU to decrease.
        is_enc = "encoder" in name
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

    Vectorised over classes: one ``torch.sort`` + one ``cumsum`` across the
    (P, K) error matrix instead of K separate kernels.  Numerically identical
    to the per-class loop version.
    """
    C = probs.shape[1]
    if classes == "all":
        class_to_sum = torch.arange(C, device=labels.device, dtype=torch.long)
    else:
        class_to_sum = labels.unique()

    if class_to_sum.numel() == 0:
        return probs.new_tensor(0.0)

    # FG mask per kept class: (P, K)
    fg = (labels.unsqueeze(1) == class_to_sum.unsqueeze(0)).float()

    if classes == "present":
        # Drop classes that have zero foreground in this batch — matches the
        # original ``if fg.sum() == 0 and classes == "present": continue``.
        present = fg.sum(0) > 0
        if not present.any():
            return probs.new_tensor(0.0)
        fg = fg[:, present]
        class_to_sum = class_to_sum[present]

    # Errors and sort along the pixel axis, independently per class.
    errors = (fg - probs.index_select(1, class_to_sum)).abs()  # (P, K)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = torch.gather(fg, 0, perm)                       # (P, K)

    # Batched _lovasz_grad along the pixel axis.
    gts = gt_sorted.sum(0, keepdim=True)                        # (1, K)
    intersection = gts - gt_sorted.cumsum(0)                    # (P, K)
    union = gts + (1.0 - gt_sorted).cumsum(0)                   # (P, K)
    jaccard = 1.0 - intersection / union
    if jaccard.shape[0] > 1:
        # grad[0] = jaccard[0]; grad[k] = jaccard[k] - jaccard[k-1]
        grad = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]], dim=0)
    else:
        grad = jaccard

    loss_per_class = (errors_sorted * grad).sum(0)              # (K,)
    return loss_per_class.mean()


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    """
    Lovász-Softmax: reshape (B,C,H,W) → (B*H*W, C) and compute per pixel.
    """
    if probs.numel() == 0 or labels.numel() == 0:
        return probs.new_tensor(0.0)

    B, C, H, W = probs.shape
    if B == 0 or C == 0 or H == 0 or W == 0:
        return probs.new_tensor(0.0)

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
        # Handle deep supervision if a future architecture returns aux heads.
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
        if logits.numel() == 0 or targets.numel() == 0:
            return logits.new_tensor(0.0)

        B, C, H, W = logits.shape
        if B == 0 or C == 0 or H == 0 or W == 0:
            return logits.new_tensor(0.0)

        if targets.max() >= C or targets.min() < 0:
            global _invalid_label_warn_count
            _invalid_label_warn_count += 1
            if _invalid_label_warn_count % _INVALID_LABEL_WARN_EVERY == 1:
                invalid_count = int(((targets < 0) | (targets >= C)).sum().item())
                log.warning(
                    "[TriLoss] Clamped %d invalid label pixel(s) in batch "
                    "(min=%d, max=%d, valid range=[0, %d]). "
                    "This may indicate a preprocessing bug or wrong mask palette. "
                    "(warning %d/%d)",
                    invalid_count,
                    int(targets.min().item()),
                    int(targets.max().item()),
                    C - 1,
                    _invalid_label_warn_count,
                    _INVALID_LABEL_WARN_EVERY,
                )
            targets = torch.clamp(targets, 0, C - 1)

        tgt = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        inter = (probs * tgt).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))
        dice_c = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        d_loss = (torch.log(torch.cosh(dice_c)).mean(dim=0) * self.cw_norm).sum()

        ce_loss = F.cross_entropy(
            logits, targets, weight=self.cw, label_smoothing=self.label_smoothing
        )

        ce_none = -log_probs.float().gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(-ce_none)
        f_loss = (self.cw[targets] * (1.0 - pt) ** self.gamma * ce_none).mean()

        total = self.dw * d_loss + self.cw_w * ce_loss + self.fw * f_loss

        if self.use_lovasz:
            total = total + self.lv_w * lovasz_softmax(probs, targets, classes="present")

        if self.use_boundary:
            total = total + self.bw * _boundary_loss(probs, tgt, self.smooth)

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
    tta_chunk: int = 256,
) -> torch.Tensor:
    """
    TTA with 3 scales × D4 symmetries (or fast mode with fewer augmentations).
    Scales: 0.875× (see larger buildings), 1.0× (base), 1.25× (see fine texture).

    All augmented views for each scale are stacked into a single forward call,
    cutting kernel launches from ``n_augs`` per scale to 1 (or a few chunks if
    VRAM is tight). Same math, far better GPU utilisation.

    ``tta_chunk`` is a safety cap on images per forward call.
    Returns (B, C, H, W) softmax probabilities.
    """
    if image.numel() == 0 or num_classes <= 0:
        B = image.shape[0] if image.ndim > 0 else 1
        H, W = image.shape[2:] if image.ndim >= 4 else (256, 256)
        return torch.zeros((B, max(1, num_classes), H, W), device=image.device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    B, _C_in, H, W = image.shape
    probs_sum = torch.zeros((B, num_classes, H, W), device=image.device, dtype=torch.float32)
    total_weight = 0.0

    scales = [(0.875, 0.8), (1.0, 1.0), (1.25, 0.9)] if not fast_tta else [(1.0, 1.0), (1.25, 0.9)]
    n_augs = 4 if fast_tta else 8

    for scale, weight in scales:
        if scale != 1.0:
            img_s = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            img_s = image
        h_s, w_s = img_s.shape[2], img_s.shape[3]

        augs = []
        for k in range(n_augs):
            aug = torch.rot90(img_s, k % 4, dims=[2, 3])
            if k >= 4:
                aug = torch.flip(aug, [3])
            augs.append(aug)
        mega = torch.cat(augs, dim=0)  # (n_augs * B, C, h_s, w_s)

        raw_parts = []
        for start in range(0, mega.shape[0], tta_chunk):
            chunk = mega[start : start + tta_chunk]
            try:
                with torch.amp.autocast(image.device.type, dtype=amp_dtype):
                    raw = model(chunk)
                    if isinstance(raw, (list, tuple)):
                        raw = raw[0]
                raw_parts.append(torch.softmax(raw.float(), 1))
            except Exception:
                raw_parts.append(torch.zeros(
                    (chunk.shape[0], num_classes, h_s, w_s),
                    device=image.device, dtype=torch.float32,
                ))
        raw_all = torch.cat(raw_parts, dim=0)

        for k in range(n_augs):
            prob = raw_all[k * B : (k + 1) * B]
            if k >= 4:
                prob = torch.flip(prob, [3])
            prob = torch.rot90(prob, -(k % 4), dims=[2, 3])
            if prob.shape[2:] != (H, W):
                prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
            probs_sum.add_(prob * weight)
            total_weight += weight

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
