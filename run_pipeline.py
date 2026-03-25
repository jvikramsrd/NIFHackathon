# run_pipeline.py -- SVAMITVA dataset edition
#
# USAGE (use forward slashes or double-backslash in Windows paths):
#
#   Preprocess:
#     python run_pipeline.py --mode preprocess --data_root "C:/Users/Dell/Downloads/dataset"
#
#   Train all stages:
#     python run_pipeline.py --mode train_all
#
#   Evaluate on validation split:
#     python run_pipeline.py --mode evaluate
#
#   Infer on a new village TIF:
#     python run_pipeline.py --mode infer --tif "C:/path/VILLAGE.tif" --out ./outputs/village
#
#   All steps in one go:
#     python run_pipeline.py --mode all --data_root "C:/Users/Dell/Downloads/dataset"

# ── Windows multiprocessing guard — MUST be first ───────────────────────────
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as CFG

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(data_root: str):
    from data.preprocessing import preprocess_folder

    data_root_path = Path(data_root)

    # Find all sub-folders that contain rasters (cg/, pb/, or more)
    # Also handle the case where data_root itself contains rasters
    candidates = [data_root_path] + [d for d in data_root_path.iterdir() if d.is_dir()]

    RASTER_EXTS = {".tif", ".tiff", ".ecw", ".img"}
    folders_with_rasters = []
    for d in candidates:
        has_raster = any(
            f.suffix.lower() in RASTER_EXTS for f in d.iterdir() if f.is_file()
        )
        if has_raster:
            folders_with_rasters.append(d)

    if not folders_with_rasters:
        print(f"\n[ERROR] No raster files found under {data_root_path}")
        print("  Expected structure:")
        print("    dataset/cg/*.tif  +  *.shp")
        print("    dataset/pb/*.tif  +  *.shp")
        return

    print(f"\nFolders to process: {[f.name for f in folders_with_rasters]}")

    all_summaries = []
    for folder in folders_with_rasters:
        summary = preprocess_folder(str(folder), CFG)
        all_summaries.append(summary)

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("  PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    total_patches = sum(s["patches"] for s in all_summaries)
    total_crops = sum(s["crops"] for s in all_summaries)
    total_infra = sum(s["infra"] for s in all_summaries)
    total_failed = sum(s["failed"] for s in all_summaries)

    for s in all_summaries:
        folder_name = Path(s["folder"]).name
        print(f"\n  {folder_name}/")
        print(f"    Rasters processed : {s['rasters']}")
        print(f"    Rasters failed    : {s['failed']}")
        print(f"    Patches           : {s['patches']}")
        print(f"    Building crops    : {s['crops']}")
        print(f"    Infra objects     : {s['infra']}")

    print("\n  TOTALS:")
    print(f"    Patches           : {total_patches}")
    print(f"    Building crops    : {total_crops}")
    print(f"    Infra objects     : {total_infra}")
    print(f"    Failed rasters    : {total_failed}")
    print("\n  Output dirs:")
    print(f"    Patches    → {CFG.PATCH_DIR}")
    print(f"    Crops      → {CFG.CROP_DIR}")
    print(f"    YOLO       → {CFG.YOLO_DIR}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────


def train_all():
    from train.train_stage1 import train_stage1
    from train.train_stage2 import train_stage2a, train_stage2b
    from utils.hardware import clear_cuda_cache

    _header("STAGE 1 — Semantic Segmentation  (Swin-B UNet++)")
    train_stage1()
    clear_cuda_cache()  # free Swin-B weights before loading ConvNeXt

    _header("STAGE 2A — Rooftop Classifier  (ConvNeXt-Base)")
    train_stage2a()
    clear_cuda_cache()  # free ConvNeXt weights before YOLOv8

    _header("STAGE 2B — Infrastructure Detector  (YOLOv8-l)")
    train_stage2b()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────


def evaluate():
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import split_dataset
    from models.stage1_segmentation import Stage1Module
    from train.train_stage1 import _validate
    from utils.hardware import get_amp_context, setup
    from utils.metrics import SegmentationMetrics

    device = setup()
    amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)
    cfg = CFG.STAGE1
    ckpt_p = CFG.CKPT_DIR / "stage1_best.pth"

    if not ckpt_p.exists():
        print("[ERROR] No checkpoint found. Run --mode train_all first.")
        return

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    module = Stage1Module(cfg).to(device)
    module.load_state_dict(ckpt["state_dict"], strict=False)

    _, val_ds = split_dataset(
        str(CFG.PATCH_DIR),
        str(CFG.MASK_DIR),
        float(cfg["val_fraction"]),  # type: ignore
        int(cfg["seed"]),  # type: ignore
        int(cfg["num_classes"]),  # type: ignore
        int(cfg["patch_size"]),  # type: ignore
    )
    loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    metrics = SegmentationMetrics(int(cfg["num_classes"]), cfg["class_names"])  # type: ignore
    miou, _ = _validate(module, loader, device, metrics, amp_ctx)
    print(metrics.summary())
    print(f"\nCheckpoint mIoU : {ckpt['val_miou']:.4f}")
    print(f"Re-eval mIoU    : {miou:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# INFER
# ─────────────────────────────────────────────────────────────────────────────


def infer(tif_path: str, out_dir: str):
    from inference.pipeline import GeoIntelPipeline

    pipe = GeoIntelPipeline(
        str(CFG.CKPT_DIR / "stage1_best.pth"),
        str(CFG.CKPT_DIR / "stage2a_best.pth"),
        str(CFG.CKPT_DIR / "stage2b_yolov8l" / "weights" / "best.pt"),
    )
    pipe.run(tif_path, out_dir)


# ─────────────────────────────────────────────────────────────────────────────


def _header(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "preprocess",
            "train_stage1",
            "train_stage2",
            "train_all",
            "evaluate",
            "infer",
            "all",
        ],
    )
    ap.add_argument(
        "--data_root",
        default="./dataset",
        help="Root of dataset folder (contains cg/ and pb/ subfolders)",
    )
    ap.add_argument("--tif", default=None, help="Test raster for --mode infer")
    ap.add_argument("--out", default="./outputs/test", help="Output dir for infer")
    args = ap.parse_args()

    if args.mode == "preprocess":
        preprocess(args.data_root)
    elif args.mode == "train_stage1":
        from train.train_stage1 import train_stage1

        train_stage1()
    elif args.mode == "train_stage2":
        from train.train_stage2 import train_stage2a, train_stage2b

        train_stage2a()
        train_stage2b()
    elif args.mode == "train_all":
        train_all()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "infer":
        assert args.tif, "--tif is required for infer mode"
        infer(args.tif, args.out)
    elif args.mode == "all":
        preprocess(args.data_root)
        train_all()
        evaluate()
