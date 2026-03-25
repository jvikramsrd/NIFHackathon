"""
run_stage2b.py  —  Standalone Stage 2B: Infrastructure Detection
═══════════════════════════════════════════════════════════════════
Extracts infrastructure data from shapefiles → YOLO format,
then trains YOLOv8x with all the accuracy improvements:

  • Class-specific bounding box sizes (transformer=60px, tank=50px, well=25px)
  • Object-centered tiles (no edge cropping)
  • Negative tile sampling for false-positive reduction
  • Copy-paste augmentation, cosine LR, early stopping

Usage:
  python run_stage2b.py                        # extract + train
  python run_stage2b.py --extract-only         # extract data only
  python run_stage2b.py --train-only           # train only (data must exist)
  python run_stage2b.py --no-resume            # train from scratch
  python run_stage2b.py --data-dir C:\\path\\to  # override dataset folder
"""

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from utils.hardware import setup, vram_stats, clear_cuda_cache


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_infra_data(data_dirs: list):
    """
    Scan data folders for rasters + Utility shapefiles,
    extract infrastructure tiles with YOLO labels.
    """
    from data.preprocessing import scan_folder, _extract_infra_streaming

    cfg2b = CFG.STAGE2B
    out_img_dir = str(CFG.YOLO_DIR / "images")
    out_lbl_dir = str(CFG.YOLO_DIR / "labels")

    total_infra = 0

    for folder in data_dirs:
        folder = Path(folder)
        if not folder.exists():
            print(f"  [SKIP] Folder not found: {folder}")
            continue

        print(f"\n{'='*60}")
        print(f"  Scanning: {folder}")
        print(f"{'='*60}")

        rasters, shps = scan_folder(str(folder))
        if not rasters:
            print(f"  No rasters found in {folder}")
            continue

        # Find utility shapefiles
        shp_by_stem = {s.stem.lower().rstrip("_"): s for s in shps}
        utility_shps = [s for k, s in shp_by_stem.items() if k.startswith("utility")]

        if not utility_shps:
            print(f"  No Utility shapefiles found in {folder}")
            continue

        print(f"  Utility SHPs: {[s.name for s in utility_shps]}")

        for raster_path in rasters:
            print(f"\n  Processing: {raster_path.name}")
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
                )
                total_infra += n
                print(f"    → {n} infrastructure objects extracted")
            except Exception as e:
                print(f"    [ERROR] {e}")
                continue

            gc.collect()

    # Print summary
    img_count = len(list((CFG.YOLO_DIR / "images").glob("*.png")))
    lbl_count = len(list((CFG.YOLO_DIR / "labels").glob("*.txt")))
    neg_count = len(list((CFG.YOLO_DIR / "images").glob("infra_neg_*.png")))

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total infrastructure objects: {total_infra}")
    print(f"  Tile images: {img_count}  ({neg_count} negative)")
    print(f"  Label files: {lbl_count}")
    print(f"  Output: {CFG.YOLO_DIR}")
    print(f"{'='*60}")

    return total_infra


# ─────────────────────────────────────────────────────────────────────────────
# 2.  YOLO YAML + TRAIN/VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def prepare_yolo_dataset():
    """
    Split extracted tiles into train/val and write data.yaml for YOLO.
    """
    import yaml

    cfg2b = CFG.STAGE2B
    yolo_dir = CFG.YOLO_DIR
    imgs_dir = yolo_dir / "images"

    all_imgs = sorted(imgs_dir.glob("*.png"))
    if len(all_imgs) < 2:
        print(f"  [ERROR] Need at least 2 images, found {len(all_imgs)} in {imgs_dir}")
        return ""

    # 15% validation split
    val_frac = 0.15
    n_val = max(1, int(len(all_imgs) * val_frac))
    if n_val >= len(all_imgs):
        n_val = len(all_imgs) - 1

    train_imgs = all_imgs[n_val:]
    val_imgs = all_imgs[:n_val]

    print(f"  Train: {len(train_imgs)} tiles  |  Val: {len(val_imgs)} tiles")

    # Create train/val directory structure
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        split_img_dir = yolo_dir / split / "images"
        split_lbl_dir = yolo_dir / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p in imgs:
            lbl_p = yolo_dir / "labels" / img_p.with_suffix(".txt").name
            dst_i = split_img_dir / img_p.name
            dst_l = split_lbl_dir / lbl_p.name

            # Copy if not already there
            if not dst_i.exists():
                try:
                    dst_i.symlink_to(img_p.resolve())
                except OSError:
                    shutil.copy2(img_p, dst_i)

            if lbl_p.exists() and not dst_l.exists():
                try:
                    dst_l.symlink_to(lbl_p.resolve())
                except OSError:
                    shutil.copy2(lbl_p, dst_l)

    # Write data.yaml
    data = {
        "path": str(yolo_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": cfg2b["num_classes"],
        "names": cfg2b["class_names"],
    }
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))

    print(f"  YOLO config: {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_stage2b(data_yaml: str, resume: bool = True):
    """
    Train YOLOv8x infrastructure detector with tuned hyperparams.
    """
    from models.stage2_models import InfrastructureDetector

    cfg2b = CFG.STAGE2B
    variant = cfg2b["model_variant"]

    print(f"\n{'='*60}")
    print(f"  STAGE 2B TRAINING: {variant}")
    print(f"  Image size: {cfg2b['img_size']}px  |  Batch: {cfg2b['batch_size']}")
    print(f"  LR: {cfg2b['lr0']} → {cfg2b['lrf']}  |  Epochs: {cfg2b['epochs']}")
    print(f"  Warmup: {cfg2b.get('warmup_epochs', 3)} epochs  |  Patience: {cfg2b.get('patience', 20)}")
    print(f"  Cosine LR: {cfg2b.get('cos_lr', True)}  |  Copy-paste: {cfg2b.get('copy_paste', 0)}")
    print(f"  {vram_stats()}")
    print(f"{'='*60}")

    run_dir = CFG.CKPT_DIR / f"stage2b_{variant}"
    best_ckpt = run_dir / "weights" / "best.pt"
    last_ckpt = run_dir / "weights" / "last.pt"

    # Pick the best available checkpoint for fine-tuning
    finetune_from = None
    if resume:
        if best_ckpt.exists():
            finetune_from = str(best_ckpt)
        elif last_ckpt.exists():
            finetune_from = str(last_ckpt)

    if finetune_from:
        # IMPORTANT: Do NOT use YOLO's `resume=True` — it reloads the old
        # training config (including data=coco.yaml) and tries to download COCO.
        # Instead, load the checkpoint weights into a fresh model and train
        # on OUR data.yaml with OUR hyperparams.
        print(f"  Fine-tuning from: {finetune_from}")
        from ultralytics import YOLO
        model = YOLO(finetune_from)
        model.train(
            data=data_yaml,
            epochs=cfg2b["epochs"],
            imgsz=cfg2b["img_size"],
            batch=cfg2b["batch_size"],
            device="0",
            project=str(CFG.CKPT_DIR),
            name=f"stage2b_{variant}",
            exist_ok=True,
            pretrained=True,
            lr0=float(cfg2b.get("lr0", 0.001)),
            lrf=float(cfg2b.get("lrf", 0.01)),
            warmup_epochs=float(cfg2b.get("warmup_epochs", 3)),
            patience=int(cfg2b.get("patience", 20)),
            cos_lr=bool(cfg2b.get("cos_lr", True)),
            copy_paste=float(cfg2b.get("copy_paste", 0)),
            cache=cfg2b.get("cache", "disk"),
        )
        detector = InfrastructureDetector(cfg2b, str(CFG.CKPT_DIR))
    else:
        if last_ckpt.exists() or best_ckpt.exists():
            print(f"  [INFO] Previous checkpoint found but --no-resume specified, training from scratch")
        detector = InfrastructureDetector(cfg2b, str(CFG.CKPT_DIR))
        detector.train(data_yaml, device="0")

    # Show results
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n  Best model saved: {best_pt}")
        print(f"  Model size: {best_pt.stat().st_size / 1e6:.1f} MB")

    print(f"\nStage 2B training complete.")
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2B: Infrastructure Detection (extract + train)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stage2b.py                                 # full pipeline
  python run_stage2b.py --extract-only                  # data extraction only
  python run_stage2b.py --train-only                    # train only
  python run_stage2b.py --no-resume                     # fresh training
  python run_stage2b.py --data-dir C:\\dataset\\cg        # custom data folder
  python run_stage2b.py --data-dir C:\\cg --data-dir C:\\pb  # multiple folders
        """,
    )
    parser.add_argument(
        "--data-dir", action="append", default=None,
        help="Path to dataset folder(s) containing rasters + Utility SHPs. "
             "Can be specified multiple times. Default: scans dataset/ subfolders.",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only extract data, don't train.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train (assumes data already extracted).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Train from scratch, don't resume from checkpoint.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing YOLO data before extraction (fresh start).",
    )

    args = parser.parse_args()

    # Setup hardware
    device = setup(verbose=True)
    print(f"\n  Device: {device}")

    # Determine data directories
    if args.data_dir:
        data_dirs = args.data_dir
    else:
        # Auto-discover: scan all subdirectories of dataset/
        data_root = CFG.DATA_ROOT
        if data_root.exists():
            data_dirs = [
                str(d) for d in sorted(data_root.iterdir())
                if d.is_dir() and d.name not in {
                    "patches", "patch_masks", "building_crops",
                    "yolo_infra", "masks",
                }
            ]
        else:
            data_dirs = []

        if not data_dirs:
            print(f"\n  [ERROR] No dataset folders found in {data_root}")
            print(f"  Use --data-dir to specify the path to your data folder")
            sys.exit(1)

    print(f"\n  Data folders: {data_dirs}")

    # Clean if requested
    if args.clean:
        yolo_dir = CFG.YOLO_DIR
        for sub in ["images", "labels", "train", "val"]:
            d = yolo_dir / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        print("  Cleaned existing YOLO data")

    # ── Extract ──────────────────────────────────────────────────────────
    if not args.train_only:
        print("\n" + "─" * 60)
        print("  PHASE 1: Data Extraction")
        print("─" * 60)
        n = extract_infra_data(data_dirs)
        if n == 0:
            print("\n  [ERROR] No infrastructure objects found. Check your Utility shapefiles.")
            if not args.extract_only:
                print("  Skipping training.")
            sys.exit(1)

    if args.extract_only:
        print("\n  Done (extract-only mode).")
        return

    # ── Prepare YOLO dataset ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 2: Preparing YOLO Dataset")
    print("─" * 60)
    clear_cuda_cache()
    data_yaml = prepare_yolo_dataset()
    if not data_yaml:
        print("  [ERROR] Failed to create YOLO dataset")
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 3: Training YOLOv8x")
    print("─" * 60)
    train_stage2b(data_yaml, resume=not args.no_resume)

    print("\n  All done! ✓")


if __name__ == "__main__":
    main()
