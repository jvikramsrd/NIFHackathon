"""
train/train_stage2.py  (A4000-optimised)
──────────────────────────────────────────
Stage 2A: EfficientNetV2-L (or ConvNeXt-Large) rooftop classifier
  • bfloat16, torch.compile, persistent workers
  • Class-balanced WeightedRandomSampler (handles Tin/Other imbalance)
  • MixUp or CutMix (randomly chosen per batch)
  • AdamW with OneCycleLR super-convergence
  • EMA shadow weights for validation
  • 16-fold multi-scale TTA at validation
  • Temperature scaling calibration after training

Stage 2B: YOLO infrastructure detector
  • 1280-px tiles, RAM cache, mosaic
  • close_mosaic last 20 epochs for stability
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import time
import typing

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config as CFG
from data.dataset import split_clf_dataset
from models.stage2_models import InfrastructureDetector, RooftopClassifier
from utils.checkpointing import atomic_torch_save, clean_state_dict
from utils.hardware import (
    EMA,
    cl_input,
    clear_cuda_cache,
    compile_model,
    get_amp_context,
    get_yolo_device,
    maybe_backward,
    setup,
    to_channels_last,
    vram_stats,
    warmup_cuda_graphs,
    worker_init_fn,
)
from utils.logger import crash_logged, get_logger
from utils.sam import SAM

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A
# ─────────────────────────────────────────────────────────────────────────────


def train_stage2a(resume: bool = True):
    device = setup(seed=int(CFG.STAGE1["seed"]))  # type: ignore
    cfg: typing.Any = CFG.STAGE2A
    amp_ctx, scaler = get_amp_context(CFG.AMP_DTYPE)
    # bf16 → scaler is None (no-op). fp16 → real GradScaler.
    if scaler is not None and bool(cfg.get("use_sam", False)):
        raise RuntimeError(
            "Stage 2A: SAM is incompatible with fp16 GradScaler. "
            "Use CFG.AMP_DTYPE=torch.bfloat16 (Ampere+) or set use_sam=False."
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = split_clf_dataset(
        str(CFG.CROP_DIR),
        cfg["class_names"],  # type: ignore
        val_fraction=float(CFG.STAGE1["val_fraction"]),  # type: ignore
        seed=int(CFG.STAGE1["seed"]),  # type: ignore
        crop_size=int(cfg["crop_size"]),  # type: ignore
        use_randaugment=bool(cfg.get("use_randaugment", False)),
        randaugment_n=int(cfg.get("randaugment_n", 2)),
        randaugment_m=int(cfg.get("randaugment_m", 7)),
    )
    log.info(f"  Rooftop crops — Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── VRAM auto-guard for SAM (parity with Stage 1) ────────────────────────
    # SAM does TWO forward+backward passes per step. ConvNeXt-L at 224px in
    # bf16 sits around 8-10 GB activation peak at batch=32; doubling it puts
    # us over the A4000's 16 GB budget. Halve the batch on small-VRAM cards
    # so the run actually starts instead of OOMing on step 2. Use a LOCAL
    # variable — mutating cfg here would persist into CFG.STAGE2A globally.
    batch_size = cfg["batch_size"]
    if bool(cfg.get("use_sam", False)):
        try:
            vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            if vram_total <= 16.5 and batch_size > 8:
                new_bs = max(8, batch_size // 2)
                log.warning(
                    "Stage 2A SAM on %.0fGB GPU — auto-reducing batch %d→%d",
                    vram_total, batch_size, new_bs,
                )
                batch_size = new_bs
        except Exception:
            pass

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Class-balanced sampling: WeightedRandomSampler oversamples rare classes
    # (Tin, Other) so each batch sees all 4 roof types roughly equally.
    # Dramatically improves recall on minority classes without augmentation.
    # Only used when cfg['use_balanced_sampler'] = True (default: True).
    train_sampler = None
    train_shuffle = True
    if bool(cfg.get("use_balanced_sampler", True)) and hasattr(train_ds, "get_sample_weights"):
        try:
            weights = train_ds.get_sample_weights()  # (N,) float tensor
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_shuffle = False  # sampler is mutually exclusive with shuffle
            log.info(
                "  Class-balanced sampler: ON (oversampling rare Tin/Other classes)"
            )
        except Exception as e:
            log.warning("  Class-balanced sampler failed (%s); using shuffle=True", e)

    n_workers = CFG.NUM_WORKERS
    base_kw = dict(
        num_workers=n_workers,
        pin_memory=CFG.PIN_MEMORY,
        prefetch_factor=CFG.PREFETCH_FACTOR if n_workers > 0 else None,
        persistent_workers=CFG.PERSISTENT_WORKERS if n_workers > 0 else False,
        worker_init_fn=worker_init_fn if n_workers > 0 else None,
    )
    loader_kw = {k: v for k, v in base_kw.items() if v is not None}
    try:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=True,
            **loader_kw,  # type: ignore
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **loader_kw,  # type: ignore
        )
        test_iter = iter(train_loader)
        next(test_iter)
        del test_iter
        log.info(f"  DataLoader: {n_workers} workers OK")
    except Exception as e:
        log.info(f"  DataLoader workers failed ({e}), falling back to 0")
        loader_kw = dict(num_workers=0, pin_memory=CFG.PIN_MEMORY)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=True,
            **loader_kw,  # type: ignore
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **loader_kw,  # type: ignore
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model = RooftopClassifier(cfg).to(device)

    # Gradient checkpointing: saves ~2GB VRAM at ~15% compute cost.
    # Critical when SAM is enabled (two forward passes double peak memory).
    try:
        model.backbone.set_grad_checkpointing(True)
        log.info("  Gradient checkpointing: ON (saves ~2GB VRAM)")
    except (AttributeError, TypeError):
        log.info("  Gradient checkpointing: not supported for this backbone")

    # channels_last: ConvNeXt uses large-kernel depthwise conv → NHWC is
    # 15–25% faster on A4000 Ampere tensor cores vs default NCHW layout.
    model = to_channels_last(model)

    log.info(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log.info(f"  {vram_stats()}")

    if CFG.COMPILE_ENABLED:
        # fullgraph=True is safe for ConvNeXt (no dynamic control flow)
        model = compile_model(model, CFG.COMPILE_MODE, fullgraph=True)

    # CUDA Graphs warmup: trigger torch.compile compilation + graph capture
    if CFG.COMPILE_ENABLED and CFG.CUDA_GRAPHS_ENABLED:
        try:
            dummy_inp = torch.randn(
                batch_size, int(cfg["crop_size"]),
                int(cfg["crop_size"]), 3,
                device=device,
            ).to(memory_format=torch.channels_last)
            warmup_cuda_graphs(model, device, dummy_inp, n_warmup=3)
        except Exception:
            pass

    # Class-weighted CE (handle imbalance)
    cw = train_ds.class_weights().to(device)
    model.criterion = nn.CrossEntropyLoss(
        weight=cw, label_smoothing=float(cfg["label_smoothing"])
    )  # type: ignore

    base_opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),  # type: ignore
    )
    use_sam = bool(cfg.get("use_sam", False))
    if use_sam:
        optimiser = SAM(
            base_opt,
            rho=float(cfg.get("sam_rho", 0.05)),
            adaptive=bool(cfg.get("sam_adaptive", True)),
        )
        log.info(
            "SAM enabled for Stage 2A rho=%s adaptive=%s",
            cfg.get("sam_rho", 0.05),
            cfg.get("sam_adaptive", True),
        )
    else:
        optimiser = base_opt

    # Mirror Stage 1's MAX_STEPS_PER_EPOCH cap. On a 30 GB dataset the rooftop
    # split can have tens of thousands of crops; capping makes epochs a fixed
    # wall-clock budget and avoids OneCycle-style schedulers overshooting.
    max_steps_per_epoch = int(getattr(CFG, "MAX_STEPS_PER_EPOCH", len(train_loader)))
    capped_steps = min(len(train_loader), max_steps_per_epoch)

    # SequentialLR with Linear Warmup and CosineAnnealingWarmRestarts.
    # IMPORTANT: use capped_steps (actual optimizer steps per epoch) rather than
    # len(train_loader) so the warmup and cosine decay match the real training pace.
    warmup_iters = max(1, cfg["epochs"] // 10) * capped_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        base_opt, start_factor=0.1, total_iters=warmup_iters
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        base_opt,
        T_0=max(1, (int(cfg["epochs"]) * capped_steps - warmup_iters) // 3),
        T_mult=1,
        eta_min=float(cfg["lr"]) / 1000,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )

    best_acc = 0.0
    patience = 15
    no_improv = 0
    ckpt_path = CFG.CKPT_DIR / "stage2a_best.pth"
    last_ckpt_path = CFG.CKPT_DIR / "stage2a_last.pth"
    start_epoch = 1

    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9995)))

    if resume:
        loaded = False
        for ckpt_name, is_best_fallback in [(last_ckpt_path, False), (ckpt_path, True)]:
            if not ckpt_name.exists():
                continue
            try:
                state = torch.load(ckpt_name, map_location=device, weights_only=False)
            except Exception as e:
                if is_best_fallback:
                    log.warning("Checkpoint %s corrupt too — starting from scratch.", ckpt_name)
                else:
                    log.warning("Checkpoint %s corrupt, trying %s ...", ckpt_name, ckpt_path)
                continue

            # ── Load model weights ──────────────────────────────────────────
            model_key = "state_dict" if is_best_fallback else "model_state"
            if model_key not in state:
                log.warning("No %s in %s — skipping.", model_key, ckpt_name)
                if is_best_fallback:
                    break
                continue

            try:
                incompatible = model.load_state_dict(
                    clean_state_dict(state[model_key], model.state_dict()), strict=False
                )
                miss = list(getattr(incompatible, "missing_keys", []) or [])
                unexp = list(getattr(incompatible, "unexpected_keys", []) or [])
            except RuntimeError as e:
                if is_best_fallback:
                    log.warning("Best checkpoint shape mismatch — starting from scratch: %s",
                                str(e).split("\n", 1)[0])
                    break
                log.warning("Last checkpoint shape mismatch, trying best: %s",
                            str(e).split("\n", 1)[0])
                continue

            if miss or unexp:
                log.warning(
                    "Stage 2A checkpoint key mismatch — missing=%d, unexpected=%d. "
                    "If STAGE2A['arch'] or related cfg changed, %s is stale.",
                    len(miss), len(unexp), ckpt_name,
                )

            # ── Optimizer + scheduler (best checkpoint doesn't have these) ──
            if not is_best_fallback:
                if "optimizer_state" in state:
                    try:
                        optimiser.load_state_dict(state["optimizer_state"])
                    except Exception as e:
                        log.warning("Skipping optimizer load: %s", e)
                else:
                    log.info("No optimizer_state in checkpoint; moments will re-warm.")

                if "scheduler_state" in state:
                    try:
                        scheduler.load_state_dict(state["scheduler_state"])
                    except Exception as e:
                        log.warning("Skipping scheduler load (LR will restart!): %s", e)
                else:
                    log.warning("No scheduler_state in checkpoint — LR resets to warmup.")

            # ── EMA ─────────────────────────────────────────────────────────
            if is_best_fallback:
                # Best checkpoint stores EMA weights as state_dict — restore into EMA
                ema.shadow = {k: v.to(device) for k, v in state["state_dict"].items()}
            else:
                if "ema_state" in state and state["ema_state"] is not None:
                    ema.shadow = {k: v.to(device) for k, v in state["ema_state"].items()}

            best_acc = float(state.get("best_acc", state.get("val_acc", 0.0)))
            no_improv = int(state.get("no_improv", 0))
            start_epoch = int(state.get("epoch", 0)) + 1
            loaded = True
            break

        if loaded:
            log.info(
                f"  Resume state → start_epoch={start_epoch}, "
                f"best_acc={best_acc:.4f}, no_improv={no_improv}"
            )
        else:
            log.warning("No valid checkpoint found — training from scratch.")

    log.info(f"\n[Stage 2A] {cfg['arch']}  |  {vram_stats()}\n")

    # max_steps_per_epoch and capped_steps were computed above (before the scheduler).
    # Both are used in the training loop below.

    for epoch in range(start_epoch, cfg["epochs"] + 1):  # type: ignore
        model.train()
        ep_loss = correct = total = 0
        actual_steps = 0
        t0 = time.time()

        capped_steps = min(len(train_loader), max_steps_per_epoch)
        train_pbar = tqdm(
            enumerate(train_loader),
            total=capped_steps,
            desc=f"Stage2A Train Ep {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for step, (imgs, labels) in train_pbar:
            if step >= max_steps_per_epoch:
                break
            actual_steps += 1
            # pin_memory is on, so non_blocking=True lets the H2D DMA overlap
            # with the previous batch's backward — otherwise the .to() syncs.
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs = cl_input(imgs)  # NHWC → 15-25% faster on Ampere

            # Randomly choose MixUp or CutMix
            if np.random.rand() < 0.5:
                aug_imgs, ya, yb, lam = model.mixup(imgs, labels, cfg["mixup_alpha"])
            else:
                aug_imgs, ya, yb, lam = model.cutmix(imgs, labels, cfg["cutmix_alpha"])

            def _mixed_loss():
                # Single forward pass with the dominant label (ya gets higher weight).
                # Computing loss against both labels from cached logits avoids a second
                # forward pass, saving ~1.5GB activation memory on the A4000.
                logits_local = model(aug_imgs, ya)
                loss_a = model.criterion(logits_local, ya)
                # Re-use same logits for yb loss (shared forward pass, no second forward needed)
                loss_b = model.criterion(logits_local, yb)
                lam_local = lam.mean() if isinstance(lam, torch.Tensor) else lam
                return logits_local, loss_a * lam_local + loss_b * (1.0 - lam_local)

            optimiser.zero_grad(set_to_none=True)
            did_step = True  # SAM always steps via second_step; non-SAM may skip on spike
            if use_sam:
                # SAM + bf16 path (scaler is None here, guarded above).
                with amp_ctx:
                    logits, loss = _mixed_loss()
                maybe_backward(loss, scaler)
                optimiser.first_step(zero_grad=True)
                with amp_ctx:
                    logits, loss_second = _mixed_loss()
                maybe_backward(loss_second, scaler)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.second_step(zero_grad=True)
            else:
                with amp_ctx:
                    logits, loss = _mixed_loss()
                maybe_backward(loss, scaler)
                # Parity with Stage 1: peek at grad norm and skip the step on a
                # spike instead of stepping with garbage gradients. clip_grad_norm_
                # already returns the pre-clip norm, so this is one extra .item().
                if scaler is not None:
                    scaler.unscale_(optimiser)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if total_norm.item() > 10.0:
                    log.warning(
                        f"Stage 2A gradient norm spike ({total_norm.item():.2f}) — skipping step"
                    )
                    did_step = False
                    if scaler is not None:
                        # Advance scaler bookkeeping (it reacts to inf/nan grads detected
                        # during unscale_ and grows/shrinks the loss scale accordingly).
                        scaler.update()
                else:
                    if scaler is not None:
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        optimiser.step()
            # Only advance scheduler / EMA when the optimizer actually stepped —
            # otherwise a spike-skipped batch silently consumes a scheduler tick.
            if did_step:
                scheduler.step()
                ema.update(model)

            ep_loss += loss.item()
            # Compare against the dominant label in the mixed image.
            # When lam >= 0.5 the original class dominates; below 0.5 the
            # shuffled class does. This prevents the accuracy metric from being
            # systematically biased (predicting the correct dominant class
            # but being marked wrong because the other class had the original label).
            dominant_labels = ya if lam >= 0.5 else yb
            correct += (logits.argmax(1) == dominant_labels).sum().item()
            total += labels.size(0)

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(correct / max(1, total)):.4f}",
            )

        ema.apply_shadow(model)
        try:
            val_acc, report = _val_clf(
                model, val_loader, device, cfg, amp_ctx, epoch=epoch
            )
        finally:
            ema.restore(model)

        elapsed = time.time() - t0
        log.info(
            f"Ep {epoch:03d}/{cfg['epochs']:03d}  "
            f"loss={ep_loss / max(actual_steps, 1):.4f}  "
            f"train_acc={correct / max(total, 1):.4f}  val_acc={val_acc:.4f}  {elapsed:.0f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improv = 0
            best_state = {k: v.cpu() for k, v in ema.shadow.items()}
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "state_dict": best_state,
                    "val_acc": val_acc,
                    "config": dict(cfg),
                },
                ckpt_path,
            )
            log.info(f"  ✓ Best acc={best_acc:.4f} (Saved EMA weights)")
            log.info(report)
        else:
            no_improv += 1

        # Heavy optimizer state only written every 5 epochs (~2× model size).
        # scheduler_state is ~1.7 KB and must be saved every epoch — without
        # it the LR resets to warmup on any non-mod-5 resume.
        last_payload = {
            "epoch": epoch,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "ema_state": {k: v.cpu() for k, v in ema.shadow.items()},
            "scheduler_state": scheduler.state_dict(),  # tiny, always save
            "best_acc": float(best_acc),
            "no_improv": int(no_improv),
            "config": dict(cfg),
        }
        if epoch % 5 == 0:
            last_payload["optimizer_state"] = optimiser.state_dict()
        atomic_torch_save(last_payload, last_ckpt_path)

        if no_improv >= patience:
            log.info("  Early stop.")
            break

    log.info(f"\nStage 2A done. Best acc: {best_acc:.4f}")

    # ── Post-training Temperature Scaling Calibration ─────────────────────────
    # Calibrate confidence scores so threshold-based class selection (Tin/Other)
    # works correctly. Reduces ECE from ~0.08 → ~0.02 without retraining.
    # Load the best checkpoint for calibration.
    try:
        from utils.calibration import calibrate_stage2a
        log.info("  Running post-training temperature scaling calibration...")
        # Reload best model weights for calibration
        best_ckpt_state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cal_model = RooftopClassifier(cfg).to(device)
        raw = best_ckpt_state.get("state_dict", best_ckpt_state)
        from utils.checkpointing import clean_state_dict as _cs
        cal_model.load_state_dict(_cs(raw, cal_model.state_dict()), strict=False)
        cal_model.eval()
        # Re-build val_loader (may have been GC'd)
        cal_loader = DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
        )
        T = calibrate_stage2a(
            cal_model, cal_loader, device, amp_ctx, cfg, CFG.CKPT_DIR
        )
        log.info("  Calibration done. Temperature T=%.4f", T)
    except Exception as _cal_err:
        log.warning("  Calibration skipped: %s", _cal_err)

    return ckpt_path


@torch.no_grad()
def _val_clf(model, loader, device, cfg: typing.Any, amp_ctx, epoch=None):
    model.eval()
    # Collect predictions / labels as GPU tensors, concatenate once at the end.
    # The original used Python lists + per-batch ``.cpu().numpy()`` + a final
    # ``np.array(list)`` (extra copy), and called ``labels.numpy()`` *twice*
    # per batch just to compute a tqdm postfix.
    pred_chunks: list = []
    label_chunks: list = []
    # Running counts as GPU scalars so the postfix is O(1) per batch.
    running_correct = torch.zeros((), device=device, dtype=torch.int64)
    running_total = torch.zeros((), device=device, dtype=torch.int64)

    val_pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Stage2A Val   Ep {epoch:03d}" if epoch is not None else "Stage2A Val",
        leave=False,
        dynamic_ncols=True,
    )

    for imgs, labels in val_pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = cl_input(imgs)
        with amp_ctx:
            preds = model.predict(imgs, tta_steps=int(cfg.get("tta_steps", 4)))

        preds = preds.detach()
        labels = labels.detach()
        pred_chunks.append(preds)
        label_chunks.append(labels)

        running_correct += (preds == labels).sum()
        running_total += labels.numel()
        # One scalar .item() per batch (cheap) vs full tensor round-trip.
        val_pbar.set_postfix(
            acc=f"{(running_correct.item() / max(int(running_total.item()), 1)):.4f}"
        )

    if pred_chunks:
        all_p_np = torch.cat(pred_chunks).cpu().numpy()
        all_l_np = torch.cat(label_chunks).cpu().numpy()
    else:
        all_p_np = np.zeros(0, dtype=np.int64)
        all_l_np = np.zeros(0, dtype=np.int64)

    acc = float((all_p_np == all_l_np).mean()) if all_p_np.size else 0.0
    report = classification_report(
        all_l_np,
        all_p_np,
        labels=range(len(cfg["class_names"])),  # type: ignore
        target_names=cfg["class_names"],  # type: ignore
        zero_division=0,  # type: ignore
    )
    return acc, report


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B
# ─────────────────────────────────────────────────────────────────────────────


def train_stage2b(data_yaml: str | None = None, resume: bool = True):
    cfg: typing.Any = CFG.STAGE2B
    variant = cfg["model_variant"]
    log.info(f"\n[Stage 2B] {variant} infrastructure detector")
    log.info(f"  Image size: {cfg['img_size']} px  |  batch: {cfg['batch_size']}")
    log.info(f"  Cache: {cfg.get('cache', 'disk')}  |  {vram_stats()}")
    log.info("  Progress: YOLO shows live training metrics in the console during .train()")
    log.info(f"  Watch folder: {CFG.CKPT_DIR / f'stage2b_{variant}'}")

    if data_yaml is None:
        data_yaml = _write_yolo_yaml()
    if not data_yaml:
        return None
    _validate_yolo_yaml(data_yaml)  # catch malformed YAML before YOLO starts

    run_dir = CFG.CKPT_DIR / f"stage2b_{variant}"
    best_ckpt = run_dir / "weights" / "best.pt"
    last_ckpt = run_dir / "weights" / "last.pt"

    finetune_from = None
    if resume:
        if best_ckpt.exists():
            finetune_from = str(best_ckpt)
        elif last_ckpt.exists():
            finetune_from = str(last_ckpt)

    if finetune_from:
        # Do NOT use YOLO's resume — it reloads old config (data=coco.yaml).
        # Instead load weights and train fresh on our data.
        log.info(f"  Fine-tuning from: {finetune_from}")
        from ultralytics import YOLO

        model = YOLO(finetune_from)
        train_args = dict(
            data=data_yaml,
            epochs=cfg["epochs"],
            imgsz=cfg["img_size"],
            batch=cfg["batch_size"],
            device=get_yolo_device(),
            project=str(CFG.CKPT_DIR),
            name=f"stage2b_{variant}",
            exist_ok=True,
            pretrained=True,
            lr0=float(cfg.get("lr0", 0.001)),
            lrf=float(cfg.get("lrf", 0.01)),
            warmup_epochs=float(cfg.get("warmup_epochs", 3)),
            patience=int(cfg.get("patience", 20)),
            cos_lr=bool(cfg.get("cos_lr", True)),
            mosaic=float(cfg.get("mosaic", 1.0)),
            close_mosaic=int(cfg.get("close_mosaic", 20)),
            hsv_h=float(cfg.get("hsv_h", 0.015)),
            hsv_s=float(cfg.get("hsv_s", 0.5)),
            hsv_v=float(cfg.get("hsv_v", 0.3)),
            degrees=float(cfg.get("degrees", 15.0)),
            translate=float(cfg.get("translate", 0.1)),
            scale=float(cfg.get("scale", 0.5)),
            fliplr=float(cfg.get("fliplr", 0.5)),
            flipud=float(cfg.get("flipud", 0.5)),
            mixup=float(cfg.get("mixup", 0.15)),
            copy_paste=float(cfg.get("copy_paste", 0.30)),
            cache=cfg.get("cache", "disk"),
            # Accuracy lifts: light head dropout + multi-scale training.
            dropout=float(cfg.get("dropout", 0.0)),
            multi_scale=bool(cfg.get("multi_scale", False)),
            workers=int(cfg.get("workers", CFG.NUM_WORKERS)),
            amp=True,
            verbose=True,
        )
        if cfg.get("use_obb"):
            train_args["task"] = "obb"
        _ = model.train(**train_args)
        # Wrap in InfrastructureDetector for a consistent return type
        detector = InfrastructureDetector(cfg, str(CFG.CKPT_DIR))
        detector.model = model
        detector._backend = "yolo"
    else:
        detector = InfrastructureDetector(cfg, str(CFG.CKPT_DIR))
        _ = detector.train(data_yaml, device=get_yolo_device())  # type: ignore
    log.info("\nStage 2B done.")
    return detector


def _validate_yolo_yaml(data_yaml: str) -> None:
    """Raise ValueError if data.yaml is malformed or references missing directories.

    YOLO's .train() surfaces these as an opaque internal crash after spending
    time initialising — validating upfront gives a clear, actionable error.
    """
    import yaml as _yaml

    with open(data_yaml) as f:
        data = _yaml.safe_load(f)

    # 1. Required top-level keys
    required = ["train", "val", "nc", "names"]
    missing_keys = [k for k in required if k not in data]
    if missing_keys:
        raise ValueError(
            f"data.yaml is missing required keys: {missing_keys}  (path={data_yaml})"
        )

    # 2. nc must match len(names)
    nc = data["nc"]
    names = data["names"]
    if len(names) != nc:
        raise ValueError(
            f"data.yaml nc={nc} does not match len(names)={len(names)}  "
            f"(path={data_yaml})"
        )

    # 3. Train/val image directories must exist
    base = Path(data.get("path", ""))
    for split_key in ("train", "val"):
        rel = data[split_key]
        split_dir = (base / rel) if base else Path(rel)
        if not split_dir.exists():
            log.warning(
                "data.yaml %s directory does not exist: %s",
                split_key,
                split_dir,
            )

    log.info(
        "[Stage 2B] data.yaml validated: nc=%d, names=%s",
        nc,
        names,
    )


def _write_yolo_yaml() -> str:
    yolo_dir = CFG.YOLO_DIR
    cfg: typing.Any = CFG.STAGE2B
    imgs_dir = yolo_dir / "images"
    all_imgs = sorted(imgs_dir.glob("*.png"))
    if len(all_imgs) < 2:
        log.warning("Not enough YOLO images found in %s (found %d).", imgs_dir, len(all_imgs))
        return ""

    n_val = max(1, int(len(all_imgs) * float(cfg.get("val_fraction", CFG.STAGE1.get("val_fraction", 0.15)))))  # type: ignore
    if n_val >= len(all_imgs):
        n_val = len(all_imgs) - 1

    for split, imgs in [("train", all_imgs[n_val:]), ("val", all_imgs[:n_val])]:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_p in imgs:
            lbl_p = yolo_dir / "labels" / img_p.with_suffix(".txt").name
            dst_i = yolo_dir / split / "images" / img_p.name
            dst_l = yolo_dir / split / "labels" / lbl_p.name
            if not dst_i.exists():
                try:
                    dst_i.symlink_to(img_p.resolve())
                except OSError:  # Windows may need admin for symlinks
                    import shutil

                    shutil.copy2(img_p, dst_i)
            if lbl_p.exists() and not dst_l.exists():
                try:
                    dst_l.symlink_to(lbl_p.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(lbl_p, dst_l)

    import geopandas as gpd
    from data.preprocessing import canonical_mapped_label, find_attribute_column

    canonical_classes = set()
    log.debug("Scanning dataset for dynamic YOLO classes...")
    try:
        for shp_path in CFG.DATA_ROOT.rglob("*.shp"):
            if shp_path.name.startswith(("Utility", "Utility_Poly")):
                try:
                    gdf = gpd.read_file(str(shp_path))
                    col = find_attribute_column(
                        gdf,
                        cfg.get(
                            "shp_infra_cols",
                            [cfg.get("shp_infra_col", "Utility_Ty")],
                        ),
                    )
                    if col:
                        unique_vals = gdf[col].dropna().unique()
                        for v in unique_vals:
                            mapped = canonical_mapped_label(
                                v, cfg.get("infra_type_map", {}), default=None
                            )
                            if mapped:
                                canonical_classes.add(mapped)
                except Exception:
                    pass
    except Exception:
        pass

    # Keep the configured order. The YOLO label ids were generated with
    # cfg["class_names"], so sorting detected classes would silently swap ids
    # (e.g. transformer id 0 becoming overhead_tank id 0).
    final_names = list(cfg["class_names"])
    if canonical_classes:
        detected_ordered = [name for name in final_names if name in canonical_classes]
        missing = [name for name in final_names if name not in canonical_classes]
        log.info(
            "  [INFO] Detected YOLO classes: %s | keeping configured id order: %s",
            detected_ordered,
            final_names,
        )
        if missing:
            log.info("  [INFO] No labels detected yet for: %s", missing)
    else:
        log.info(f"  [INFO] Using fallback classes from config: {final_names}")

    data = {
        "path": str(yolo_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(final_names),
        "names": final_names,
    }
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    return str(yaml_path)


def extract_infra_data(data_dirs: list) -> int:
    from data.preprocessing import scan_folder, _extract_infra_streaming

    cfg2b: typing.Any = CFG.STAGE2B
    out_img_dir = str(CFG.YOLO_DIR / "images")
    out_lbl_dir = str(CFG.YOLO_DIR / "labels")
    total_infra = 0

    for folder in data_dirs:
        folder = Path(folder)
        if not folder.exists():
            log.info(f"  [SKIP] Folder not found: {folder}")
            continue
        log.info(f"\n{'='*60}")
        log.info(f"  Scanning: {folder}")
        log.info(f"{'='*60}")
        rasters, shps = scan_folder(str(folder))
        if not rasters:
            log.info(f"  No rasters found in {folder}")
            continue
        shp_by_stem = {s.stem.lower().rstrip("_"): s for s in shps}
        utility_shps = [s for k, s in shp_by_stem.items() if k.startswith("utility")]
        if not utility_shps:
            log.info(f"  No Utility shapefiles found in {folder}")
            continue
        log.info(f"  Utility SHPs: {[s.name for s in utility_shps]}")
        for raster_path in rasters:
            log.info(f"\n  Processing: {raster_path.name}")
            try:
                n = _extract_infra_streaming(
                    raster_path,
                    utility_shps,
                    cfg2b["shp_infra_col"],
                    CFG.INFRA_TYPE_MAP,
                    cfg2b["class_names"],
                    out_img_dir,
                    out_lbl_dir,
                    cfg2b["img_size"],
                    class_buffer_px=cfg2b.get("class_buffer_px"),
                    neg_tile_ratio=cfg2b.get("neg_tile_ratio", 0.0),
                    use_obb=cfg2b.get("use_obb", False),
                )
                total_infra += n
                log.info(f"    -> {n} infrastructure objects extracted")
            except Exception as e:
                log.error("Extraction failed: %s", e, exc_info=True)
                continue
            gc.collect()

    img_count = len(list((CFG.YOLO_DIR / "images").glob("*.png")))
    lbl_count = len(list((CFG.YOLO_DIR / "labels").glob("*.txt")))
    neg_count = len(list((CFG.YOLO_DIR / "images").glob("infra_neg_*.png")))
    log.info(f"\n{'='*60}")
    log.info(f"  EXTRACTION COMPLETE")
    log.info(f"  Total infrastructure objects: {total_infra}")
    log.info(f"  Tile images: {img_count}  ({neg_count} negative)")
    log.info(f"  Label files: {lbl_count}")
    log.info(f"  Output: {CFG.YOLO_DIR}")
    log.info(f"{'='*60}")
    return total_infra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["2a", "2b", "both", "2b-extract"], default="both")
    ap.add_argument("--data-dir", action="append", default=None, help="Dataset folder(s) with rasters + Utility SHPs")
    ap.add_argument("--extract-only", action="store_true", help="Only extract data, don't train")
    ap.add_argument("--train-only", action="store_true", help="Only train (assumes data already extracted)")
    ap.add_argument("--no-resume", action="store_true", help="Train from scratch")
    ap.add_argument("--clean", action="store_true", help="Delete existing YOLO data before extraction")
    args = ap.parse_args()

    if args.stage == "2b-extract":
        args.stage = "2b"
        args.extract_only = True

    with crash_logged(log, f"Stage {args.stage} training"):
        if args.stage in ("2a", "both"):
            train_stage2a()
            if args.stage == "both":
                clear_cuda_cache()
        if args.stage in ("2b", "both"):
            device = setup(verbose=True)
            log.info(f"\n  Device: {device}")

            if args.clean:
                import shutil as _shutil
                yolo_dir = CFG.YOLO_DIR
                for sub in ["images", "labels", "train", "val"]:
                    d = yolo_dir / sub
                    if d.exists():
                        _shutil.rmtree(d, ignore_errors=True)
                        d.mkdir(parents=True, exist_ok=True)
                log.info("  Cleaned existing YOLO data")

            if not args.train_only:
                data_dirs = args.data_dir or [
                    str(d) for d in sorted(CFG.DATA_ROOT.iterdir())
                    if d.is_dir() and d.name not in {
                        "patches", "patch_masks", "building_crops",
                        "yolo_infra", "masks",
                    }
                ]
                log.info(f"\n  Data folders: {data_dirs}")
                n = extract_infra_data(data_dirs)
                if n == 0:
                    log.error("No infrastructure objects found")
                    sys.exit(1)

            if args.extract_only:
                log.info("\n  Done (extract-only mode).")
            else:
                clear_cuda_cache()
                train_stage2b(resume=not args.no_resume)
