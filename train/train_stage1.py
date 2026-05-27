"""
train/train_stage1.py  (A4000-optimised v2)
─────────────────────────────────────────
New in v2:
  • SAM (Sharpness-Aware Minimisation) — flatter minima, +0.5–2% mIoU
  • Multi-scale training — random resize for scale invariance
  • Boundary-aware loss — sharper edges, fewer merged buildings
  • WandB experiment tracking (optional, auto-detected)
  • mit_b4 encoder (production speed/accuracy balance)
  • Structured logging via utils.logger
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import math
import time
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as CFG
from data.dataset import split_dataset
from models.stage1_segmentation import Stage1Module
from utils.checkpointing import atomic_torch_save, clean_state_dict
from utils.ddp import is_main_process, make_loader, set_epoch, setup_ddp, wrap_ddp
from utils.hardware import (
    EMA, compile_model, get_amp_context, maybe_backward, maybe_step, setup,
    to_channels_last, vram_stats, warmup_cuda_graphs, worker_init_fn,
)
from utils.logger import crash_logged, get_logger
from utils.metrics import SegmentationMetrics
from utils.metrics import SegmentationMetrics

log = get_logger(__name__)

def train_stage1(resume: bool = True):
    cfg = cast(Dict[str, Any], CFG.STAGE1)
    # DDP-aware setup: no-op when WORLD_SIZE=1 (single GPU / no torchrun)
    ddp = setup_ddp(seed=cfg["seed"])
    device = ddp.device if ddp.enabled else setup(seed=cfg["seed"])
    amp_ctx, scaler = get_amp_context(CFG.AMP_DTYPE)
    # scaler is None for bf16 (no-op fast path), a real GradScaler for fp16.
    # SAM's two-step protocol is incompatible with GradScaler (it would need a
    # mid-step unscale + re-scale that the public API doesn't expose).
    if scaler is not None and bool(cfg.get("use_sam", False)):
        raise RuntimeError(
            "SAM is not compatible with fp16 GradScaler. "
            "Switch CFG.AMP_DTYPE to torch.bfloat16 (Ampere+) or disable SAM."
        )

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

    # ── Auto-detect VRAM and tune batch size if needed ───────────────────────
    encoder = cfg["encoder"]
    try:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if vram_gb < 12 and batch_size > 4:
            old_bs = batch_size
            batch_size = 4
            log.warning(
                f"VRAM={vram_gb:.1f}GB — reducing Stage 1 batch "
                f"{old_bs}→{batch_size} while keeping encoder={encoder}"
            )
    except Exception:
        pass

    # ── Model ────────────────────────────────────────────────────────────────
    # Single build with the configured Unet/MiT encoder baked into the config copy.
    module = Stage1Module({**cfg, "encoder": encoder}).to(device)
    # channels_last (NHWC): 15-30% faster on Ampere tensor cores
    module = to_channels_last(module)

    log.info(f"Encoder : {encoder}")
    log.info(f"Params  : {sum(p.numel() for p in module.parameters()) / 1e6:.1f}M")

    try:
        module.model.encoder.set_grad_checkpointing(True)
        log.info("Gradient checkpointing: ON")
    except Exception:
        log.info("Gradient checkpointing: not supported (OK)")

    if CFG.COMPILE_ENABLED:
        module.model = compile_model(module.model, CFG.COMPILE_MODE)

    # CUDA Graphs warmup: trigger torch.compile compilation + graph capture
    # before the main loop so the first real step isn't slowed down.
    if CFG.COMPILE_ENABLED and CFG.CUDA_GRAPHS_ENABLED:
        try:
            dummy_inp = torch.randn(
                int(cfg["batch_size"]), int(cfg["in_channels"]),
                int(cfg["patch_size"]), int(cfg["patch_size"]),
                device=device,
            )
            warmup_cuda_graphs(module.model, device, dummy_inp, n_warmup=3)
        except Exception:
            pass

    # Wrap in DDP — no-op when world_size=1
    module.model = wrap_ddp(module.model, ddp)

    ema = EMA(module, decay=cfg["ema_decay"]) if cfg.get("use_ema") else None

    # ── Optimiser ────────────────────────────────────────────────────────────
    param_groups = module.parameter_groups()
    optimiser = torch.optim.AdamW(param_groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    grad_accum = int(cfg["grad_accum"])

    steps_per_ep = math.ceil(min(len(train_loader), getattr(CFG, "MAX_STEPS_PER_EPOCH", len(train_loader))) / grad_accum)
    max_lrs = [g.get("lr", cfg["lr"]) for g in param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=max_lrs, epochs=cfg["epochs"], steps_per_epoch=steps_per_ep,
        pct_start=0.15, div_factor=25, final_div_factor=1e4,
    )

    # SWA
    swa_model, swa_scheduler = None, None
    swa_start = int(cfg["epochs"] * cfg["swa_start_frac"])
    if cfg.get("use_swa"):
        swa_model = torch.optim.swa_utils.AveragedModel(module)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimiser, swa_lr=cfg["swa_lr"], anneal_epochs=5
        )

    # ── Metrics + checkpointing ──────────────────────────────────────────────
    metrics = SegmentationMetrics(cfg["num_classes"], cfg["class_names"])
    best_miou = 0.0
    patience = 18
    no_improv = 0
    ckpt_path = CFG.CKPT_DIR / "stage1_best.pth"
    last_ckpt_path = CFG.CKPT_DIR / "stage1_last.pth"
    start_epoch = 1

    if resume and last_ckpt_path.exists():
        if is_main_process(ddp):
            log.info(f"Resuming Stage 1 from: {last_ckpt_path}")
        start_epoch, best_miou, no_improv = _load_training_state(
            module=module, optimiser=optimiser, scheduler=scheduler,
            ema=ema, ckpt_path=last_ckpt_path, device=device,
        )
        if is_main_process(ddp):
            log.info(f"Resume: start_epoch={start_epoch}, best_mIoU={best_miou:.4f}, no_improv={no_improv}")

    log.info(f"[Stage 1] Starting training — {cfg['epochs']} epochs")
    log.info(f"  Effective batch: {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    log.info(f"  {vram_stats()}")

    max_steps_per_epoch = int(getattr(CFG, "MAX_STEPS_PER_EPOCH", len(train_loader)))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        module.train()
        ep_loss = 0.0
        t0 = time.time()

        # Tell DistributedSampler which epoch we're in (ensures per-epoch shuffle)
        set_epoch(train_loader, epoch)

        # Initialise gradients at epoch start
        optimiser.zero_grad(set_to_none=True)

        capped_steps = min(len(train_loader), max_steps_per_epoch)
        train_pbar = tqdm(enumerate(train_loader), total=capped_steps,
                          desc=f"Train Ep {epoch:03d}", leave=False, dynamic_ncols=True)
        actual_steps = 0

        for step, (imgs, masks) in train_pbar:
            if step >= max_steps_per_epoch:
                break
            actual_steps += 1
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            def _forward():
                with amp_ctx:
                    logits = module(imgs)
                    return module.loss(logits, masks) / grad_accum

            loss = _forward()
            maybe_backward(loss, scaler)

            if (step + 1) % grad_accum == 0:
                # PEEK at gradient norm before stepping to skip on spike
                if scaler is not None:
                    scaler.unscale_(optimiser)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(), 1.0,
                )
                if total_norm.item() > 10.0:
                    log.warning(
                        f"Gradient norm spike ({total_norm.item():.2f}) — skipping step"
                    )
                    if scaler is not None:
                        scaler.update()
                else:
                    if scaler is not None:
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        optimiser.step()
                    try:
                        scheduler.step()
                    except ValueError:
                        pass
                    if ema:
                        ema.update(module)
                optimiser.zero_grad(set_to_none=True)

            ep_loss += loss.item() * grad_accum
            train_pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        # Flush remaining gradients if steps run is not a multiple of grad_accum
        if (actual_steps % grad_accum) != 0:
            # The loss was divided by grad_accum in _forward(), but we only accumulated
            # (actual_steps % grad_accum) steps. Rescale gradients to correct magnitude.
            scale_factor = grad_accum / (actual_steps % grad_accum)
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale_factor)
            
            if scaler is not None:
                scaler.unscale_(optimiser)
            total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            if total_norm.item() <= 10.0:
                if scaler is not None:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                try:
                    scheduler.step()
                except ValueError:
                    pass
                if ema:
                    ema.update(module)
            elif scaler is not None:
                scaler.update()
            optimiser.zero_grad(set_to_none=True)

        ep_loss /= max(actual_steps, 1)

        if ema:
            ema.apply_shadow(module)
        val_tta_every = max(1, int(getattr(CFG, "VAL_TTA_EVERY", 1)))
        use_val_tta = (epoch % val_tta_every) == 0
        try:
            val_miou, val_loss = _validate(
                module,
                val_loader,
                device,
                metrics,
                amp_ctx,
                epoch=epoch,
                use_tta=use_val_tta,
            )
        finally:
            # Critical: if _validate raises (OOM, CUDA error, etc.), the model
            # would be left with EMA shadow weights swapped in. The next training
            # step would then `optimiser.step()` on EMA weights and feed those
            # right back into the next ema.update — silently corrupting training.
            if ema:
                ema.restore(module)

        # TTA mega-batches (n_augs × B per scale) blow up the cached allocator
        # on the A4000's 16 GB heap. Release reserved-but-unused blocks before
        # the next training epoch so peak fragmentation doesn't compound.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if swa_model and epoch >= swa_start:
            swa_model.update_parameters(module)
            if swa_scheduler:
                swa_scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimiser.param_groups[-1]["lr"]

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
            # gloo (used on Windows DDP) doesn't support broadcasting CUDA tensors
            stop_tensor = torch.tensor(int(do_early_stop), device="cpu")
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



@torch.no_grad()
def _validate(module, loader, device, metrics, amp_ctx, epoch=None, use_tta=True):
    """Validation pass.

    ``use_tta=True`` runs the same batched fast TTA used at inference (2 scales
    × 4 folds, one batched forward per scale via tta_predict). This makes the
    val mIoU representative of deployed-model mIoU, so the best-checkpoint
    selector picks a model that actually performs better at inference — a real
    accuracy win at the cost of ~2-3× validation time per epoch (training is
    unaffected; val is a small fraction of epoch time).
    """
    module.eval()
    metrics.reset()
    total_loss = 0.0
    # Cached once — criterion.cw is a constant buffer, unaffected by EMA swap.
    class_weights = module.criterion.cw
    max_val_steps = min(
        len(loader),
        int(getattr(CFG, "MAX_VAL_STEPS", len(loader))),
    )
    val_iter = tqdm(loader, total=max_val_steps,
                    desc=f"Val   Ep {epoch:03d}" if epoch else "Validation",
                    leave=False, dynamic_ncols=True)
    # GPU-side confusion matrix accumulator — see prior comment.
    nc = metrics.num_classes
    gpu_cm = torch.zeros((nc, nc), dtype=torch.int64, device=device)
    batches_seen = 0
    for step, (imgs, masks) in enumerate(val_iter):
        if step >= max_val_steps:
            break
        batches_seen += 1
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_ctx:
            raw = module(imgs)
            logits = raw[0] if isinstance(raw, (list, tuple)) else raw
        # Loss is for logging only; CE on the single forward is fine.
        loss = F.cross_entropy(logits.float(), masks, weight=class_weights)

        if use_tta:
            # Predictions use the batched fast TTA (1 forward per scale).
            preds = module.predict(
                imgs, use_tta=True, fast_tta=True, amp_dtype=CFG.AMP_DTYPE,
            )
        else:
            preds = logits.float().argmax(1)

        p = preds.reshape(-1).to(torch.int64)
        t = masks.reshape(-1).to(torch.int64)
        valid = (t >= 0) & (t < nc)
        if valid.any():
            p = p[valid]
            t = t[valid]
            idx = nc * t + p
            gpu_cm += torch.bincount(idx, minlength=nc * nc).reshape(nc, nc)

        total_loss += loss.item()
        val_iter.set_postfix(loss=f"{loss.item():.4f}")

    # One small transfer instead of B×(B,H,W) per batch
    metrics._conf_mat = gpu_cm.cpu().numpy()

    res = metrics.compute()
    miou = res["mean_iou"]
    for name, iou in zip(metrics.class_names, res["class_iou"]):
        log.info(f"    {name:12s} IoU={iou:.3f}")
    return miou, total_loss / max(batches_seen, 1)


def _save_best(module, ema, epoch, miou, cfg, path):
    if ema:
        weights = {k: v.detach().cpu() for k, v in ema.shadow.items()}
    else:
        weights = module.state_dict()
    atomic_torch_save({"epoch": epoch, "state_dict": weights, "val_miou": miou, "config": dict(cfg)}, path)


_OPTIM_SAVE_EVERY = 5  # write the heavy optimizer/scheduler state every N epochs


def _save_last(module, ema, optimiser, scheduler, epoch, best_miou, no_improv, cfg, path):
    # AdamW moments re-warm up in a few steps so the heavy optimizer state
    # (~2× model size) is saved only every _OPTIM_SAVE_EVERY epochs to avoid
    # dominating checkpoint I/O.
    # ⚠ The scheduler_state is ~1.7 KB regardless of model size, but losing
    # it means OneCycleLR restarts at step 0 on resume — the LR would be the
    # warmup LR instead of the actual mid-training value. ALWAYS save it.
    model_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    ema_state = {k: v.detach().cpu() for k, v in ema.shadow.items()} if ema else None
    payload = {
        "epoch": epoch, "model_state": model_state, "ema_state": ema_state,
        "scheduler_state": scheduler.state_dict(),  # tiny, always save
        "best_miou": float(best_miou), "no_improv": int(no_improv), "config": dict(cfg),
    }
    if epoch % _OPTIM_SAVE_EVERY == 0:
        payload["optimizer_state"] = optimiser.state_dict()
    atomic_torch_save(payload, path)


def _load_training_state(module, optimiser, scheduler, ema, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state") or ckpt.get("state_dict", {})

    # strict=False is needed so DDP-wrapped checkpoints (prefixed with
    # "module.") still load into a non-DDP run, but it also silently drops
    # mismatched KEYS after an arch change. Shape mismatches still raise even
    # with strict=False, so we catch those explicitly and produce the same
    # clear message instead of an opaque PyTorch traceback.
    try:
        model_state = clean_state_dict(model_state, module.state_dict())
        incompatible = module.load_state_dict(model_state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    except RuntimeError as e:
        log.error(
            "Checkpoint shape mismatch at %s — almost certainly a stale "
            "checkpoint from a different STAGE1['arch'] or ['encoder']. "
            "Delete the checkpoint to retrain from scratch, or restore the "
            "original architecture. Underlying error:\n  %s",
            ckpt_path, str(e).split("\n", 1)[0],
        )
        raise
    if missing or unexpected:
        log.warning(
            "Checkpoint key mismatch on resume — missing=%d, unexpected=%d. "
            "If you recently changed STAGE1['arch'] or ['encoder'], "
            "the old checkpoint at %s is stale and training is effectively from scratch.",
            len(missing), len(unexpected), ckpt_path,
        )
        if missing:
            log.warning("  example missing keys: %s", missing[:3])
        if unexpected:
            log.warning("  example unexpected keys: %s", unexpected[:3])

    if "optimizer_state" in ckpt:
        try:
            optimiser.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            log.warning(f"Skipping optimizer load: {e}")
    else:
        log.info(
            "No optimizer_state in checkpoint (last save was a non-mod-%d epoch). "
            "AdamW moments will re-warm over a few steps.",
            _OPTIM_SAVE_EVERY,
        )

    if "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            log.warning(f"Skipping scheduler load (LR will restart at warmup!): {e}")
    else:
        log.warning(
            "No scheduler_state in checkpoint — LR will reset to warmup. "
            "This shouldn't happen with the current _save_last; please report."
        )

    if ema and ckpt.get("ema_state") is not None:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_miou = float(ckpt.get("best_miou", ckpt.get("val_miou", 0.0)))
    no_improv = int(ckpt.get("no_improv", 0))
    return start_epoch, best_miou, no_improv


if __name__ == "__main__":
    with crash_logged(log, "Stage 1 training"):
        train_stage1()
