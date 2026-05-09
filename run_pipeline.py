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
from utils.logger import get_logger

log = get_logger(__name__)

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
        log.error("No raster files found under %s", data_root_path)
        log.info("  Expected structure:")
        log.info("    dataset/cg/*.tif  +  *.shp")
        log.info("    dataset/pb/*.tif  +  *.shp")
        return

    log.info(f"\nFolders to process: {[f.name for f in folders_with_rasters]}")

    all_summaries = []
    for folder in folders_with_rasters:
        summary = preprocess_folder(str(folder), CFG)
        all_summaries.append(summary)

    # Print overall summary
    log.info(f"\n{'=' * 60}")
    log.info("  PREPROCESSING COMPLETE")
    log.info(f"{'=' * 60}")
    total_patches = sum(int(s.get("patches", 0)) for s in all_summaries)
    total_crops = sum(int(s.get("crops", 0)) for s in all_summaries)
    total_infra = sum(int(s.get("infra", 0)) for s in all_summaries)
    total_failed = sum(int(s.get("failed", 0)) for s in all_summaries)

    for s in all_summaries:
        folder_name = Path(str(s.get("folder", ""))).name
        log.info(f"\n  {folder_name}/")
        log.info(f"    Rasters processed : {s.get('rasters', 0)}")
        log.info(f"    Rasters failed    : {s.get('failed', 0)}")
        log.info(f"    Patches           : {s.get('patches', 0)}")
        log.info(f"    Building crops    : {s.get('crops', 0)}")
        log.info(f"    Infra objects     : {s.get('infra', 0)}")

    log.info("\n  TOTALS:")
    log.info(f"    Patches           : {total_patches}")
    log.info(f"    Building crops    : {total_crops}")
    log.info(f"    Infra objects     : {total_infra}")
    log.info(f"    Failed rasters    : {total_failed}")
    log.info("\n  Output dirs:")
    log.info(f"    Patches    → {CFG.PATCH_DIR}")
    log.info(f"    Crops      → {CFG.CROP_DIR}")
    log.info(f"    YOLO       → {CFG.YOLO_DIR}")
    log.info(f"{'=' * 60}\n")


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
    clear_cuda_cache()  # free ConvNeXt weights before YOLO

    _header("STAGE 2B — Infrastructure Detector  (YOLOv9/OBB)")
    num_classes = cast(int, cfg.get("num_classes", 4))
    patch_size = cast(int, cfg.get("patch_size", 512))
    val_fraction = cast(float, cfg.get("val_fraction", 0.15))
    seed = cast(int, cfg.get("seed", 42))
    class_names = cast(List[str], cfg.get("class_names", []))

    _, val_ds = split_dataset(
        str(CFG.PATCH_DIR),
        str(CFG.MASK_DIR),
        val_fraction,
        seed,
        num_classes,
        patch_size,
    )
    loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    metrics = SegmentationMetrics(num_classes, class_names)
    miou, _ = _validate(module, loader, device, metrics, amp_ctx)
    log.info(metrics.summary())
    log.info(f"\nCheckpoint mIoU : {ckpt.get('val_miou', 0.0):.4f}")
    log.info(f"Re-eval mIoU    : {miou:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# INFER
# ─────────────────────────────────────────────────────────────────────────────


def infer(tif_path: str, out_dir: str):
    from inference.pipeline import GeoIntelPipeline

    pipe = GeoIntelPipeline(
        str(CFG.CKPT_DIR / "stage1_best.pth"),
        str(CFG.CKPT_DIR / "stage2a_best.pth"),
        str(CFG.CKPT_DIR / f"stage2b_{CFG.STAGE2B['model_variant']}" / "weights" / "best.pt"),
    )
    pipe.run(tif_path, out_dir)


# ─────────────────────────────────────────────────────────────────────────────


def _header(title: str):
    log.info(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


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

    mode = str(args.mode)
    data_root = str(args.data_root)

    if mode == "preprocess":
        preprocess(data_root)
    elif mode == "train_stage1":
        from train.train_stage1 import train_stage1

        train_stage1()
    elif mode == "train_stage2":
        from train.train_stage2 import train_stage2a, train_stage2b

        train_stage2a()
        train_stage2b()
    elif mode == "train_all":
        train_all()
    elif mode == "evaluate":
        evaluate()
    elif mode == "infer":
        assert args.tif, "--tif is required for infer mode"
        infer(str(args.tif), str(args.out))
    elif mode == "all":
        preprocess(data_root)
        train_all()
        evaluate()
