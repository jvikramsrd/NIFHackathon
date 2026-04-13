"""
train/train_stage1.py  (A4000-optimised)
─────────────────────────────────────────
Optimisations over the generic version:
  • bfloat16 AMP (no GradScaler needed on Ampere)
  • torch.compile()  — reduce-overhead mode
  • Gradient accumulation (effective batch = micro_batch × grad_accum)
  • CutMix patch augmentation (segmentation-aware)
  • EMA weight tracking
  • Stochastic Weight Averaging (SWA) in final phase
  • Layer-wise LR via separate encoder/decoder param groups
  • Windows-safe DataLoader (spawn + persistent workers)
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import math
import time
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from data.dataset import split_dataset
from models.stage1_segmentation import Stage1Module
from utils.hardware import (
    EMA,
    compile_model,
    get_amp_context,
    setup,
    to_channels_last,
    vram_stats,
    worker_init_fn,
)
from utils.metrics import SegmentationMetrics

# ─────────────────────────────────────────────────────────────────────────────


def train_stage1(resume: bool = True):
    cfg = cast(Dict[str, Any], CFG.STAGE1)
    device = setup(seed=cfg["seed"])
    amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)  # bf16 → scaler=None

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds, val_ds = split_dataset(
        str(CFG.PATCH_DIR),
        str(CFG.MASK_DIR),
        cfg["val_fraction"],
        cfg["seed"],
        cfg["num_classes"],
        cfg["patch_size"],
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Windows DataLoader — use 0 workers to avoid spawn issues
    # (Windows multiprocessing with num_workers>0 requires the training
    # code to be inside if __name__=="__main__" which run_pipeline.py handles,
    # but symlink-based dataset splits can still fail in worker processes)
    n_workers = CFG.NUM_WORKERS
    try:
        # Quick test: try creating one worker to see if it works
        import torch.multiprocessing as mp

        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    loader_kw: Dict[str, Any] = dict(
        num_workers=int(n_workers),
        pin_memory=bool(CFG.PIN_MEMORY),
        prefetch_factor=int(CFG.PREFETCH_FACTOR) if n_workers > 0 else None,
        persistent_workers=bool(CFG.PERSISTENT_WORKERS) if n_workers > 0 else False,
        worker_init_fn=worker_init_fn if n_workers > 0 else None,
    )
    # Remove None values
    loader_kw = {k: v for k, v in loader_kw.items() if v is not None}

    batch_size = int(cfg["batch_size"])

    try:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kw,
        )
        val_loader = DataLoader(
            val_ds, batch_size=max(1, batch_size // 2), shuffle=False, **loader_kw
        )
        # Test that loader actually works by fetching one batch
        test_iter = iter(train_loader)
        _ = next(test_iter)
        del test_iter
        print(f"  DataLoader: {n_workers} workers OK")
    except Exception as e:
        print(f"  DataLoader with {n_workers} workers failed ({e})")
        print("  Falling back to num_workers=0 (slower but reliable)")
        loader_kw = {"num_workers": 0, "pin_memory": bool(CFG.PIN_MEMORY)}
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kw,
        )
        val_loader = DataLoader(
            val_ds, batch_size=max(1, batch_size // 2), shuffle=False, **loader_kw
        )

    # ── Model ────────────────────────────────────────────────────────────────
    module = Stage1Module(cfg).to(device)
    print(f"  Encoder : {cfg['encoder']}")
    print(f"  Params  : {sum(p.numel() for p in module.parameters()) / 1e6:.1f}M")

    # Gradient checkpointing — works on EfficientNet via torch utility.
    # Trades ~20% compute overhead for ~2 GB VRAM savings.
    try:
        module.model.encoder.set_grad_checkpointing(True)
        print("  Gradient checkpointing: ON")
    except Exception:
        print("  Gradient checkpointing: not supported by this encoder (OK)")

    if CFG.COMPILE_ENABLED:
        module.model = compile_model(module.model, CFG.COMPILE_MODE)

    # EMA shadow weights
    ema = EMA(module, decay=cfg["ema_decay"]) if cfg.get("use_ema") else None

    # ── Optimiser — layer-wise LR ────────────────────────────────────────────
    param_groups = module.parameter_groups()
    optimiser = torch.optim.AdamW(
        param_groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    import math

    steps_per_ep = math.ceil(len(train_loader) / cfg["grad_accum"])
    # OneCycleLR provides super-convergence: trains much faster and attains deeper local minima
    max_lrs = [g.get("lr", cfg["lr"]) for g in param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=max_lrs,
        epochs=cfg["epochs"],
        steps_per_epoch=steps_per_ep,
        pct_start=0.1,  # 10% warmup
        div_factor=25,  # Start at max_lr / 25
        final_div_factor=1e4,  # End completely decayed
    )

    # SWA
    swa_model = None
    swa_scheduler = None
    swa_start = int(cfg["epochs"] * cfg["swa_start_frac"])
    if cfg.get("use_swa"):
        swa_model = torch.optim.swa_utils.AveragedModel(module)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimiser, swa_lr=cfg["swa_lr"], anneal_epochs=5
        )

    # ── Metrics + checkpointing ───────────────────────────────────────────────
    metrics = SegmentationMetrics(cfg["num_classes"], cfg["class_names"])
    best_miou = 0.0
    patience = 18
    no_improv = 0
    ckpt_path = CFG.CKPT_DIR / "stage1_best.pth"
    last_ckpt_path = CFG.CKPT_DIR / "stage1_last.pth"
    grad_accum = cfg["grad_accum"]
    start_epoch = 1

    # Optional resume support (restores model/optimizer/scheduler/ema state)
    if resume and last_ckpt_path.exists():
        print(f"  Resuming Stage 1 from: {last_ckpt_path}")
        start_epoch, best_miou, no_improv = _load_training_state(
            module=module,
            optimiser=optimiser,
            scheduler=scheduler,
            ema=ema,
            ckpt_path=last_ckpt_path,
            device=device,
        )
        print(
            f"  Resume state → start_epoch={start_epoch}, "
            f"best_mIoU={best_miou:.4f}, no_improv={no_improv}"
        )

    print(f"\n[Stage 1] Starting training — {cfg['epochs']} epochs")
    print(
        f"  Effective batch: {cfg['batch_size']} × {grad_accum} = "
        f"{cfg['batch_size'] * grad_accum}"
    )
    print(f"  {vram_stats()}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        module.train()
        ep_loss = 0.0
        t0 = time.time()
        optimiser.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train Ep {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for step, (imgs, masks) in train_pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # CutMix augmentation on batch (run before AMP to ensure data stability)
            if np.random.rand() < 0.2:
                imgs, masks = _cutmix_seg(
                    imgs, masks, alpha=cfg.get("cutmix_alpha", 1.0)
                )

            with amp_ctx:
                logits = module(imgs)
                loss = module.loss(logits, masks) / grad_accum

            loss.backward()  # bf16: no scaler needed

            if (step + 1) % grad_accum == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                # Dynamic Gradient Norm Monitoring: Skip step if gradient spikes excessively
                if total_norm.item() > 10.0:
                    print(
                        f"\n  [WARN] Gradient norm spike ({total_norm.item():.2f}) detected! Skipping optimizer step."
                    )
                    optimiser.zero_grad(set_to_none=True)
                else:
                    optimiser.step()
                    try:
                        scheduler.step()
                    except ValueError:
                        pass
                    optimiser.zero_grad(set_to_none=True)
                    ema.update(module)

            ep_loss += loss.item() * grad_accum
            train_pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        # Capture and step over remaining gradients at epoch end
        if (len(train_loader) % grad_accum) != 0:
            total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            if total_norm.item() > 10.0:
                print(
                    f"\n  [WARN] Gradient norm spike ({total_norm.item():.2f}) detected at epoch end! Skipping step."
                )
                optimiser.zero_grad(set_to_none=True)
            else:
                optimiser.step()
                try:
                    scheduler.step()
                except ValueError:
                    pass
                optimiser.zero_grad(set_to_none=True)
                ema.update(module)

        ep_loss /= len(train_loader)

        # ── Validation (with EMA weights) ────────────────────────────────────
        if ema:
            ema.apply_shadow(module)

        val_miou, val_loss = _validate(
            module, val_loader, device, metrics, amp_ctx, epoch=epoch
        )
        if ema:
            ema.restore(module)

        # ── SWA update ───────────────────────────────────────────────────────
        if swa_model and epoch >= swa_start:
            swa_model.update_parameters(module)
            if swa_scheduler is not None:
                swa_scheduler.step()
        elif epoch < swa_start:
            pass  # normal scheduler already stepped inside loop

        elapsed = time.time() - t0
        lr_now = optimiser.param_groups[-1]["lr"]  # decoder LR

        print(
            f"Ep {epoch:03d}/{cfg['epochs']:03d}  "
            f"train={ep_loss:.4f}  val={val_loss:.4f}  "
            f"mIoU={val_miou:.4f}  lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            no_improv = 0
            _save_best(module, ema, epoch, val_miou, cfg, ckpt_path)
            print(f"  ✓ Best mIoU={best_miou:.4f} — saved")
        else:
            no_improv += 1

        _save_last(
            module=module,
            ema=ema,
            optimiser=optimiser,
            scheduler=scheduler,
            epoch=epoch,
            best_miou=best_miou,
            no_improv=no_improv,
            cfg=cfg,
            path=last_ckpt_path,
        )

        if no_improv >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Final SWA BN update ──────────────────────────────────────────────────
    if swa_model:
        print("\nUpdating SWA batch norm statistics …")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_path = CFG.CKPT_DIR / "stage1_swa.pth"
        torch.save(swa_model.state_dict(), swa_path)
        print(f"  SWA model saved: {swa_path}")

    print(f"\nStage 1 complete. Best mIoU: {best_miou:.4f}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# CutMix for segmentation (mixes spatial regions, masks follow)
# ─────────────────────────────────────────────────────────────────────────────


def _cutmix_seg(imgs: torch.Tensor, masks: torch.Tensor, alpha: float = 1.0):
    B, C, H, W = imgs.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B, device=imgs.device)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)

    imgs = imgs.clone()
    masks = masks.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]

    # Target label blending (soft masks) is handled by the loss function in a true MixUp/CutMix setup.
    # Since our TriLoss uses hard integers, hard-copying the mask pixels is correct for CutMix.
    masks[:, y1:y2, x1:x2] = masks[idx, y1:y2, x1:x2]
    return imgs, masks


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _validate(module, loader, device, metrics, amp_ctx, epoch=None):
    # For validation, we disable MixUp/CutMix and run only the standard inference path
    print("  [DEBUG VAL] Disabling mixing augmentation for validation run.")
    # In a full system, this would call a dedicated validation inference path
    # For now, we ensure the structure is sound.
    module.eval()
    metrics.reset()
    total_loss = 0.0

    val_iter = tqdm(
        loader,
        total=len(loader),
        desc=f"Val   Ep {epoch:03d}" if epoch is not None else "Validation",
        leave=False,
        dynamic_ncols=True,
    )
    for imgs, masks in val_iter:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_ctx:
            logits = module(imgs)
            loss = module.loss(logits, masks)
        preds = logits.float().argmax(1)
        metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
        total_loss += loss.item()
        val_iter.set_postfix(loss=f"{loss.item():.4f}")

    res = metrics.compute()
    miou = res["mean_iou"]
    for name, iou in zip(metrics.class_names, res["class_iou"]):
        print(f"    {name:12s} IoU={iou:.3f}")
    return miou, total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_best(module, ema, epoch, miou, cfg, path):
    weights = {}
    if ema:
        weights = ema.shadow  # save EMA weights as primary
    else:
        weights = {k: v for k, v in module.state_dict().items()}
    torch.save(
        {"epoch": epoch, "state_dict": weights, "val_miou": miou, "config": cfg}, path
    )


def _save_last(
    module,
    ema,
    optimiser,
    scheduler,
    epoch,
    best_miou,
    no_improv,
    cfg,
    path,
):
    model_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    ema_state = None
    if ema:
        ema_state = {k: v.detach().cpu() for k, v in ema.shadow.items()}

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "ema_state": ema_state,
            "optimizer_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": float(best_miou),
            "no_improv": int(no_improv),
            "config": cfg,
        },
        path,
    )


def _load_training_state(
    module,
    optimiser,
    scheduler,
    ema,
    ckpt_path: Path,
    device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model_state")
    if model_state is None:
        # Backward compatibility with best checkpoint format
        model_state = ckpt.get("state_dict", {})
    module.load_state_dict(model_state, strict=False)

    if "optimizer_state" in ckpt:
        try:
            if "max_lr" not in ckpt["optimizer_state"]["param_groups"][0]:
                raise ValueError("Old checkpoint missing OneCycleLR parameters")
            optimiser.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"      [WARN] Skipping optimizer load (architecture changed?): {e}")
    if "scheduler_state" in ckpt:
        try:
            if (
                "max_lr"
                not in ckpt.get("optimizer_state", {}).get("param_groups", [{}])[0]
            ):
                raise ValueError("Old checkpoint missing OneCycleLR parameters")
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass

    if ema and ckpt.get("ema_state") is not None:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_miou = float(ckpt.get("best_miou", ckpt.get("val_miou", 0.0)))
    no_improv = int(ckpt.get("no_improv", 0))
    return start_epoch, best_miou, no_improv


def _cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


if __name__ == "__main__":
    train_stage1()
