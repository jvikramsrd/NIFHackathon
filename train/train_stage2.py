"""
train/train_stage2.py  (A4000-optimised)
──────────────────────────────────────────
Stage 2A: ConvNeXt-Large rooftop classifier
  • bfloat16, torch.compile, persistent workers
  • MixUp or CutMix (randomly chosen per batch)
  • AdamW with OneCycleLR super-convergence
  • EMA shadow weights for validation
  • 16-fold multi-scale TTA at validation

Stage 2B: YOLOv9 infrastructure detector
  • 1280-px tiles, RAM cache, mosaic
  • close_mosaic last 20 epochs for stability
"""

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import argparse
import math
import os
import time
import typing

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from data.dataset import split_clf_dataset
from models.stage2_models import InfrastructureDetector, RooftopClassifier
from utils.checkpointing import atomic_torch_save
from utils.hardware import (
    cl_input,
    compile_model,
    get_amp_context,
    setup,
    to_channels_last,
    vram_stats,
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
    amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)

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
            batch_size=int(cfg["batch_size"]),  # type: ignore
            shuffle=True,
            drop_last=True,
            **loader_kw,  # type: ignore
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["batch_size"]),
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
            batch_size=int(cfg["batch_size"]),  # type: ignore
            shuffle=True,
            drop_last=True,
            **loader_kw,  # type: ignore
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["batch_size"]),
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

    # SequentialLR with Linear Warmup and CosineAnnealingWarmRestarts
    # Allows the model to warm up linearly, then stabilize at local optima while preventing catastrophic forgetting.
    warmup_iters = max(1, int(cfg["epochs"]) // 10) * len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        base_opt, start_factor=0.1, total_iters=warmup_iters
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        base_opt,
        T_0=max(1, (int(cfg["epochs"]) * len(train_loader) - warmup_iters) // 3),
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

    from utils.hardware import EMA

    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9995)))

    if resume and last_ckpt_path.exists():
        log.info(f"  Resuming Stage 2A from: {last_ckpt_path}")
        state = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"], strict=False)
        try:
            optimiser.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
        except Exception as e:
            log.warning("Skipping optimizer load: %s", e)
        best_acc = float(state.get("best_acc", 0.0))
        no_improv = int(state.get("no_improv", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
        if "ema_state" in state and state["ema_state"] is not None:
            ema.shadow = {k: v.to(device) for k, v in state["ema_state"].items()}
        log.info(
            f"  Resume state → start_epoch={start_epoch}, "
            f"best_acc={best_acc:.4f}, no_improv={no_improv}"
        )

    log.info(f"\n[Stage 2A] {cfg['arch']}  |  {vram_stats()}\n")

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):  # type: ignore
        model.train()
        ep_loss = correct = total = 0
        t0 = time.time()

        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Stage2A Train Ep {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
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
                logits_local = model.forward_train(aug_imgs, ya)
                loss_a = model.criterion(logits_local, ya)
                # Re-use same logits for yb loss (shared forward pass, no second forward needed)
                loss_b = model.criterion(logits_local, yb)
                lam_local = lam.mean() if isinstance(lam, torch.Tensor) else lam
                return logits_local, loss_a * lam_local + loss_b * (1.0 - lam_local)

            optimiser.zero_grad(set_to_none=True)
            if use_sam:
                with amp_ctx:
                    logits, loss = _mixed_loss()
                loss.backward()
                optimiser.first_step(zero_grad=True)
                with amp_ctx:
                    logits, loss_second = _mixed_loss()
                loss_second.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.second_step(zero_grad=True)
            else:
                with amp_ctx:
                    logits, loss = _mixed_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            scheduler.step()
            ema.update(model)

            ep_loss += loss.item()
            correct += (logits.argmax(1) == ya).sum().item()
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
            f"loss={ep_loss / len(train_loader):.4f}  "
            f"train_acc={correct / total:.4f}  val_acc={val_acc:.4f}  {elapsed:.0f}s"
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
                    "config": cfg,
                },
                ckpt_path,
            )
            log.info(f"  ✓ Best acc={best_acc:.4f} (Saved EMA weights)")
            log.info(report)
        else:
            no_improv += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema_state": {k: v.cpu() for k, v in ema.shadow.items()},
                "optimizer_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": float(best_acc),
                "no_improv": int(no_improv),
                "config": cfg,
            },
            last_ckpt_path,
        )

        if no_improv >= patience:
            log.info("  Early stop.")
            break

    log.info(f"\nStage 2A done. Best acc: {best_acc:.4f}")
    return ckpt_path


@torch.no_grad()
def _val_clf(model, loader, device, cfg: typing.Any, amp_ctx, epoch=None):
    model.eval()
    all_p, all_l = [], []

    val_pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Stage2A Val   Ep {epoch:03d}" if epoch is not None else "Stage2A Val",
        leave=False,
        dynamic_ncols=True,
    )

    for imgs, labels in val_pbar:
        imgs = imgs.to(device)
        imgs = cl_input(imgs)  # keep NHWC consistent with training
        with amp_ctx:
            # Use light TTA during training (cfg['tta_steps']), full TTA only at final inference
            preds = model.predict(imgs, tta_steps=int(cfg.get("tta_steps", 4))).cpu().numpy()

        all_p.extend(preds)
        all_l.extend(labels.numpy())

        batch_acc = (preds == labels.numpy()).mean() if len(preds) > 0 else 0.0
        val_pbar.set_postfix(batch_acc=f"{batch_acc:.4f}")

    acc = (np.array(all_p) == np.array(all_l)).mean()
    report = classification_report(
        all_l,
        all_p,
        labels=range(len(cfg["class_names"])),  # type: ignore
        target_names=cfg["class_names"],  # type: ignore
        zero_division=0,  # type: ignore
    )
    return acc, report


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B
# ─────────────────────────────────────────────────────────────────────────────


def train_stage2b(resume: bool = True):
    cfg: typing.Any = CFG.STAGE2B
    variant = cfg["model_variant"]
    log.info(f"\n[Stage 2B] {variant} infrastructure detector")
    log.info(f"  Image size: {cfg['img_size']} px  |  batch: {cfg['batch_size']}")
    log.info(f"  Cache: {cfg.get('cache', 'disk')}  |  {vram_stats()}")
    log.info("  Progress: YOLO shows live training metrics in the console during .train()")
    log.info(f"  Watch folder: {CFG.CKPT_DIR / f'stage2b_{variant}'}")

    data_yaml = _write_yolo_yaml()
    if not data_yaml:
        return None

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
            device="0",
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
            flipud=float(cfg.get("flipud", 0.0)),
            mixup=float(cfg.get("mixup", 0.15)),
            copy_paste=float(cfg.get("copy_paste", 0.30)),
            cache=cfg.get("cache", "disk"),
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
        _ = detector.train(data_yaml, device="0")  # type: ignore
    log.info("\nStage 2B done.")
    return detector


def _write_yolo_yaml() -> str:
    yolo_dir = CFG.YOLO_DIR
    cfg: typing.Any = CFG.STAGE2B
    imgs_dir = yolo_dir / "images"
    all_imgs = sorted(imgs_dir.glob("*.png"))
    if len(all_imgs) < 2:
        log.warning("Not enough YOLO images found in %s (found %d).", imgs_dir, len(all_imgs))
        return ""

    n_val = max(1, int(len(all_imgs) * float(CFG.STAGE1["val_fraction"])))  # type: ignore
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

    canonical_classes = set()
    log.debug("Scanning dataset for dynamic YOLO classes...")
    try:
        for shp_path in CFG.DATA_ROOT.rglob("*.shp"):
            if shp_path.name.startswith(
                ("Utility", "Bridge", "Road", "Water", "Built")
            ):
                try:
                    gdf = gpd.read_file(str(shp_path))
                    col = cfg.get("shp_infra_col", "Utility_Ty")
                    if col in gdf.columns:
                        unique_vals = gdf[col].dropna().unique()
                        for v in unique_vals:
                            mapped = cfg.get("infra_type_map", {}).get(
                                str(v).lower(), str(v)
                            )
                            canonical_classes.add(mapped)
                except Exception:
                    pass
    except Exception:
        pass

    if canonical_classes:
        final_names = sorted(list(canonical_classes))
        log.info(f"  [INFO] Dynamically detected classes: {final_names}")
    else:
        final_names = cfg["class_names"]
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


def _cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["2a", "2b", "both"], default="both")
    args = ap.parse_args()
    with crash_logged(log, f"Stage {args.stage} training"):
        if args.stage in ("2a", "both"):
            train_stage2a()
        if args.stage in ("2b", "both"):
            train_stage2b()
