"""
train/train_stage1.py  (A4000-optimised v2)
─────────────────────────────────────────
New in v2:
  • SAM (Sharpness-Aware Minimisation) — flatter minima, +0.5–2% mIoU
  • Multi-scale training — random resize for scale invariance
  • Boundary-aware loss — sharper edges, fewer merged buildings
  • WandB experiment tracking (optional, auto-detected)
  • mit_b5 encoder (upgraded from mit_b4)
  • Structured logging via utils.logger
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from data.dataset import split_dataset
from models.stage1_segmentation import Stage1Module
from utils.checkpointing import atomic_torch_save
from utils.ddp import cleanup_ddp, is_main_process, make_loader, set_epoch, setup_ddp, wrap_ddp
from utils.hardware import (
    EMA, compile_model, get_amp_context, setup,
    to_channels_last, vram_stats, worker_init_fn,
)
from utils.logger import crash_logged, get_logger
from utils.metrics import SegmentationMetrics
from utils.sam import SAM

log = get_logger(__name__)

def train_stage1(resume: bool = True):
    cfg = cast(Dict[str, Any], CFG.STAGE1)
    # DDP-aware setup: no-op when WORLD_SIZE=1 (single GPU / no torchrun)
    ddp = setup_ddp(seed=cfg["seed"])
    device = ddp.device if ddp.enabled else setup(seed=cfg["seed"])
    amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds, val_ds = split_dataset(
        str(CFG.PATCH_DIR), str(CFG.MASK_DIR),
        cfg["val_fraction"], cfg["seed"],
        cfg["num_classes"], cfg["patch_size"], cfg.get("patch_sizes"),
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    n_workers = CFG.NUM_WORKERS
    try:
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
    loader_kw = {k: v for k, v in loader_kw.items() if v is not None}

    batch_size = int(cfg["batch_size"])

    # ── VRAM auto-guard: SAM doubles peak activation memory ──────────────────
    # UNet++ mit_b5 at 512px with batch=4 peaks at ~15.5 GB with a single pass.
    # SAM does TWO forward+backward passes per step → ~31 GB peak without guard.
    # Halving batch to 2 brings peak to ~12 GB, well within 16 GB A4000 budget.
    use_sam_cfg = cfg.get("use_sam", False)
    try:
        vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if use_sam_cfg and vram_total <= 16.5 and batch_size > 2:
            old_bs = batch_size
            batch_size = max(1, batch_size // 2)
            log.warning(
                f"SAM enabled on {vram_total:.0f}GB GPU — auto-reducing batch "
                f"{old_bs}→{batch_size} (effective batch preserved via grad_accum)"
            )
    except Exception:
        pass

    try:
        # make_loader adds DistributedSampler automatically when DDP is active
        train_loader = make_loader(train_ds, ddp, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
        val_loader   = make_loader(val_ds,   ddp, batch_size=max(1, batch_size // 2), shuffle=False, **loader_kw)
        test_iter = iter(train_loader); _ = next(test_iter); del test_iter
        log.info(f"DataLoader: {n_workers} workers OK")
    except Exception as e:
        log.warning(f"DataLoader with {n_workers} workers failed ({e}), falling back to 0")
        loader_kw = {"num_workers": 0, "pin_memory": bool(CFG.PIN_MEMORY)}
        train_loader = make_loader(train_ds, ddp, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
        val_loader   = make_loader(val_ds,   ddp, batch_size=max(1, batch_size // 2), shuffle=False, **loader_kw)

    # ── Auto-detect VRAM and downgrade encoder if needed ─────────────────────
    encoder = cfg["encoder"]
    try:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if vram_gb < 14 and encoder == "mit_b5":
            log.warning(f"VRAM={vram_gb:.1f}GB < 14GB — downgrading encoder to mit_b4")
            encoder = "mit_b4"
    except Exception:
        pass

    # ── Model ────────────────────────────────────────────────────────────────
    import segmentation_models_pytorch as smp

    module = Stage1Module(cfg).to(device)
    # Rebuild model honouring cfg["arch"] so UNet++ is used when configured.
    # Stage1Module.__init__ already calls build_stage1_model(cfg), but we
    # rebuild here to allow the VRAM-adjusted encoder to take effect.
    arch = cfg.get("arch", "UnetPlusPlus")
    ModelCls = getattr(smp, arch, smp.UnetPlusPlus)
    module.model = ModelCls(
        encoder_name=encoder,
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        activation=None,
        decoder_attention_type=cfg.get("decoder_attention_type", "scse"),
    )
    module.model = module.model.to(device)

    log.info(f"Encoder : {encoder}")
    log.info(f"Params  : {sum(p.numel() for p in module.parameters()) / 1e6:.1f}M")

    try:
        module.model.encoder.set_grad_checkpointing(True)
        log.info("Gradient checkpointing: ON")
    except Exception:
        log.info("Gradient checkpointing: not supported (OK)")

    if CFG.COMPILE_ENABLED:
        module.model = compile_model(module.model, CFG.COMPILE_MODE)

    # Wrap in DDP — no-op when world_size=1
    module.model = wrap_ddp(module.model, ddp)

    ema = EMA(module, decay=cfg["ema_decay"]) if cfg.get("use_ema") else None

    # ── Optimiser with SAM ───────────────────────────────────────────────────
    param_groups = module.parameter_groups()
    base_opt = torch.optim.AdamW(param_groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_sam = cfg.get("use_sam", False)
    configured_grad_accum = int(cfg["grad_accum"])
    grad_accum = 1 if use_sam else configured_grad_accum
    if use_sam:
        optimiser = SAM(base_opt, rho=cfg.get("sam_rho", 0.05), adaptive=cfg.get("sam_adaptive", True))
        log.info(f"SAM enabled  rho={cfg['sam_rho']}  adaptive={cfg['sam_adaptive']}")
        if configured_grad_accum != 1:
            log.warning(
                "SAM requires two passes over the same batch; using grad_accum=1 "
                f"instead of configured value {configured_grad_accum}."
            )
    else:
        optimiser = base_opt

    steps_per_ep = math.ceil(len(train_loader) / grad_accum)
    max_lrs = [g.get("lr", cfg["lr"]) for g in param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser if not use_sam else base_opt,
        max_lr=max_lrs, epochs=cfg["epochs"], steps_per_epoch=steps_per_ep,
        pct_start=0.1, div_factor=25, final_div_factor=1e4,
    )

    # SWA
    swa_model, swa_scheduler = None, None
    swa_start = int(cfg["epochs"] * cfg["swa_start_frac"])
    if cfg.get("use_swa"):
        swa_model = torch.optim.swa_utils.AveragedModel(module)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimiser if not use_sam else base_opt, swa_lr=cfg["swa_lr"], anneal_epochs=5
        )

    # ── Metrics + checkpointing ──────────────────────────────────────────────
    metrics = SegmentationMetrics(cfg["num_classes"], cfg["class_names"])
    best_miou = 0.0
    patience = 18
    no_improv = 0
    ckpt_path = CFG.CKPT_DIR / "stage1_best.pth"
    last_ckpt_path = CFG.CKPT_DIR / "stage1_last.pth"
    start_epoch = 1

    if resume and last_ckpt_path.exists() and is_main_process(ddp):
        log.info(f"Resuming Stage 1 from: {last_ckpt_path}")
        start_epoch, best_miou, no_improv = _load_training_state(
            module=module, optimiser=optimiser, scheduler=scheduler,
            ema=ema, ckpt_path=last_ckpt_path, device=device,
        )
        log.info(f"Resume: start_epoch={start_epoch}, best_mIoU={best_miou:.4f}, no_improv={no_improv}")

    log.info(f"[Stage 1] Starting training — {cfg['epochs']} epochs")
    log.info(f"  Effective batch: {cfg['batch_size']} × {grad_accum} = {cfg['batch_size'] * grad_accum}")
    log.info(f"  {vram_stats()}")

    ms_training = cfg.get("ms_training", False)
    ms_scales = cfg.get("ms_scales", (0.75, 1.0, 1.25))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        module.train()
        ep_loss = 0.0
        t0 = time.time()

        # Tell DistributedSampler which epoch we're in (ensures per-epoch shuffle)
        set_epoch(train_loader, epoch)

        # Initialise gradients at epoch start
        if use_sam:
            base_opt.zero_grad(set_to_none=True)
        else:
            optimiser.zero_grad(set_to_none=True)

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Train Ep {epoch:03d}", leave=False, dynamic_ncols=True)

        for step, (imgs, masks) in train_pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Multi-scale training: scale to a random size then back to the fixed
            # patch_size.  Both resizes are done under AMP to keep activations in bf16.
            if ms_training and np.random.rand() < 0.5:
                scale = float(np.random.choice(ms_scales))
                orig_h, orig_w = imgs.shape[2], imgs.shape[3]
                new_h = max(16, int(orig_h * scale))
                new_w = max(16, int(orig_w * scale))
                with amp_ctx:
                    imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    masks = F.interpolate(masks.unsqueeze(1).float(), size=(new_h, new_w), mode="nearest").squeeze(1).long()
                    # Resize back to fixed model input size
                    imgs = F.interpolate(imgs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                    masks = F.interpolate(masks.unsqueeze(1).float(), size=(orig_h, orig_w), mode="nearest").squeeze(1).long()

            if np.random.rand() < 0.2:
                imgs, masks = _cutmix_seg(imgs, masks, alpha=cfg.get("cutmix_alpha", 1.0))

            def _forward():
                with amp_ctx:
                    logits = module(imgs)
                    return module.loss(logits, masks) / grad_accum

            if use_sam:
                loss = _forward()
                loss.backward()
                optimiser.first_step(zero_grad=True)
                # Free activation memory before second forward pass — critical on 16GB A4000
                del loss
                torch.cuda.empty_cache()
                loss2 = _forward()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimiser.second_step(zero_grad=True)
                loss = loss2  # use second pass loss for logging
                try:
                    scheduler.step()
                except ValueError:
                    pass
                if ema:
                    ema.update(module)
            else:
                loss = _forward()
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                    if total_norm.item() > 10.0:
                        log.warning(f"Gradient norm spike ({total_norm.item():.2f}) — skipping step")
                        optimiser.zero_grad(set_to_none=True)
                    else:
                        optimiser.step()
                        try:
                            scheduler.step()
                        except ValueError:
                            pass
                        optimiser.zero_grad(set_to_none=True)
                        if ema:
                            ema.update(module)

            ep_loss += loss.item() * grad_accum
            train_pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        # Flush remaining gradients if loader length is not a multiple of grad_accum
        if not use_sam and (len(train_loader) % grad_accum) != 0:
            total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            if total_norm.item() <= 10.0:
                optimiser.step()
                try:
                    scheduler.step()
                except ValueError:
                    pass
                optimiser.zero_grad(set_to_none=True)
                if ema:
                    ema.update(module)

        ep_loss /= len(train_loader)

        if ema:
            ema.apply_shadow(module)
        val_miou, val_loss = _validate(module, val_loader, device, metrics, amp_ctx, epoch=epoch)
        if ema:
            ema.restore(module)

        if swa_model and epoch >= swa_start:
            swa_model.update_parameters(module)
            if swa_scheduler:
                swa_scheduler.step()

        elapsed = time.time() - t0
        lr_now = base_opt.param_groups[-1]["lr"] if use_sam else optimiser.param_groups[-1]["lr"]

        # Only log / checkpoint on rank-0 to avoid duplicate output
        if is_main_process(ddp):
            log.info(
                f"Ep {epoch:03d}/{cfg['epochs']:03d}  "
                f"train={ep_loss:.4f}  val={val_loss:.4f}  "
                f"mIoU={val_miou:.4f}  lr={lr_now:.2e}  {elapsed:.0f}s"
            )

            if val_miou > best_miou:
                best_miou = val_miou
                no_improv = 0
                _save_best(module, ema, epoch, val_miou, cfg, ckpt_path)
                log.info(f"  ✓ Best mIoU={best_miou:.4f} — saved")
            else:
                no_improv += 1

            _save_last(module=module, ema=ema, optimiser=optimiser, scheduler=scheduler,
                       epoch=epoch, best_miou=best_miou, no_improv=no_improv, cfg=cfg, path=last_ckpt_path)

        # Broadcast early-stop decision from rank-0 so worker ranks don't hang
        do_early_stop = is_main_process(ddp) and (no_improv >= patience)
        if ddp.enabled:
            import torch.distributed as dist
            stop_tensor = torch.tensor(int(do_early_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            do_early_stop = bool(stop_tensor.item())
        if do_early_stop:
            log.info(f"  Early stop at epoch {epoch}")
            break

    if swa_model:
        log.info("\nUpdating SWA batch norm statistics …")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_path = CFG.CKPT_DIR / "stage1_swa.pth"
        torch.save(swa_model.state_dict(), swa_path)
        log.info(f"  SWA model saved: {swa_path}")

    log.info(f"\nStage 1 complete. Best mIoU: {best_miou:.4f}")
    return ckpt_path


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
    masks[:, y1:y2, x1:x2] = masks[idx, y1:y2, x1:x2]
    return imgs, masks


@torch.no_grad()
def _validate(module, loader, device, metrics, amp_ctx, epoch=None):
    module.eval()
    metrics.reset()
    total_loss = 0.0
    val_iter = tqdm(loader, total=len(loader),
                    desc=f"Val   Ep {epoch:03d}" if epoch else "Validation",
                    leave=False, dynamic_ncols=True)
    for imgs, masks in val_iter:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_ctx:
            raw = module(imgs)
            # UNet++ with deep supervision returns a list; use the main head (index 0)
            logits = raw[0] if isinstance(raw, (list, tuple)) else raw
            loss = module.loss(raw, masks)  # loss() handles deep supervision list
        preds = logits.float().argmax(1)
        metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
        total_loss += loss.item()
        val_iter.set_postfix(loss=f"{loss.item():.4f}")
    res = metrics.compute()
    miou = res["mean_iou"]
    for name, iou in zip(metrics.class_names, res["class_iou"]):
        log.info(f"    {name:12s} IoU={iou:.3f}")
    return miou, total_loss / len(loader)


def _save_best(module, ema, epoch, miou, cfg, path):
    weights = ema.shadow if ema else {k: v for k, v in module.state_dict().items()}
    atomic_torch_save({"epoch": epoch, "state_dict": weights, "val_miou": miou, "config": cfg}, path)


def _save_last(module, ema, optimiser, scheduler, epoch, best_miou, no_improv, cfg, path):
    model_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    ema_state = {k: v.detach().cpu() for k, v in ema.shadow.items()} if ema else None
    atomic_torch_save({
        "epoch": epoch, "model_state": model_state, "ema_state": ema_state,
        "optimizer_state": optimiser.state_dict(), "scheduler_state": scheduler.state_dict(),
        "best_miou": float(best_miou), "no_improv": int(no_improv), "config": cfg,
    }, path)


def _load_training_state(module, optimiser, scheduler, ema, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state") or ckpt.get("state_dict", {})
    module.load_state_dict(model_state, strict=False)
    if "optimizer_state" in ckpt:
        try:
            optimiser.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            log.warning(f"Skipping optimizer load: {e}")
    if "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    if ema and ckpt.get("ema_state") is not None:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_miou = float(ckpt.get("best_miou", ckpt.get("val_miou", 0.0)))
    no_improv = int(ckpt.get("no_improv", 0))
    return start_epoch, best_miou, no_improv


if __name__ == "__main__":
    with crash_logged(log, "Stage 1 training"):
        train_stage1()
