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
    if not data_root_path.exists():
        log.error("Data root does not exist: %s", data_root_path)
        log.info("  Pass an existing folder via --data_root, for example:")
        log.info("    python run_pipeline.py --mode preprocess --data_root ./dataset")
        return
    if not data_root_path.is_dir():
        log.error("Data root is not a directory: %s", data_root_path)
        return

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

    _header(f"STAGE 1 — Semantic Segmentation  ({CFG.STAGE1['arch']} {CFG.STAGE1['encoder']})")
    train_stage1()
    clear_cuda_cache()

    _header(f"STAGE 2A — Rooftop Classifier  ({CFG.STAGE2A['arch']})")
    train_stage2a()
    clear_cuda_cache()

    _header("STAGE 2B — Infrastructure Detector  (YOLOv9/OBB)")
    train_stage2b()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────


def evaluate():
    import json

    import torch
    from torch.utils.data import DataLoader

    from data.dataset import split_clf_dataset, split_dataset
    from models.stage1_segmentation import Stage1Module
    from models.stage2_models import RooftopClassifier
    from train.train_stage1 import _validate
    from train.train_stage2 import _val_clf
    from utils.hardware import get_amp_context, setup
    from utils.metrics import SegmentationMetrics

    results: dict = {}
    device = setup(seed=int(CFG.STAGE1["seed"]))
    amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)
    out_path = CFG.ROOT / "outputs" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    ckpt1 = CFG.CKPT_DIR / "stage1_best.pth"
    if ckpt1.exists():
        _header("EVALUATE — Stage 1 Segmentation")
        cfg1 = CFG.STAGE1
        _, val_ds = split_dataset(
            str(CFG.PATCH_DIR), str(CFG.MASK_DIR),
            cfg1["val_fraction"], cfg1["seed"],
            cfg1["num_classes"], cfg1["patch_size"], cfg1.get("patch_sizes"),
        )
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                num_workers=0, pin_memory=True)
        module = Stage1Module(cfg1).to(device)
        ckpt = torch.load(str(ckpt1), map_location=device, weights_only=False)
        module.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        metrics = SegmentationMetrics(cfg1["num_classes"], cfg1["class_names"])
        miou, _ = _validate(module, val_loader, device, metrics, amp_ctx)
        r = metrics.compute()
        results["stage1"] = {
            "mIoU": r["mean_iou"],
            "dice": r["mean_f1"],
            "pixel_acc": r["pixel_acc"],
            "class_iou": dict(zip(cfg1["class_names"], r["class_iou"])),
        }
        log.info(metrics.summary())
        log.info(f"Stage 1 mIoU: {miou:.4f}")
    else:
        log.warning("Stage 1 checkpoint not found: %s", ckpt1)
        results["stage1"] = {"error": "checkpoint not found"}

    # ── Stage 2A ─────────────────────────────────────────────────────────────
    ckpt2a = CFG.CKPT_DIR / "stage2a_best.pth"
    if ckpt2a.exists():
        _header("EVALUATE — Stage 2A Rooftop Classification")
        cfg2a = CFG.STAGE2A
        _, val_ds2a = split_clf_dataset(
            str(CFG.CROP_DIR), cfg2a["class_names"],
            val_fraction=float(CFG.STAGE1["val_fraction"]),
            seed=int(CFG.STAGE1["seed"]),
            crop_size=int(cfg2a["crop_size"]),
        )
        val_loader2a = DataLoader(val_ds2a, batch_size=32, shuffle=False, num_workers=0)
        model2a = RooftopClassifier(cfg2a).to(device)
        ckpt = torch.load(str(ckpt2a), map_location=device, weights_only=False)
        model2a.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        acc, report = _val_clf(model2a, val_loader2a, device, cfg2a, amp_ctx)
        results["stage2a"] = {"accuracy": float(acc)}
        log.info(f"Stage 2A Accuracy: {acc:.4f}")
        log.info(report)
    else:
        log.warning("Stage 2A checkpoint not found: %s", ckpt2a)
        results["stage2a"] = {"error": "checkpoint not found"}

    # ── Stage 2B — read YOLO's own results.csv ────────────────────────────────
    variant = CFG.STAGE2B["model_variant"]
    yolo_csv = CFG.CKPT_DIR / f"stage2b_{variant}" / "results.csv"
    if yolo_csv.exists():
        import pandas as pd
        df = pd.read_csv(str(yolo_csv))
        df.columns = [c.strip() for c in df.columns]
        map50_cols = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
        if map50_cols:
            best_map50 = float(df[map50_cols[0]].max())
            results["stage2b"] = {"mAP_50": best_map50}
            log.info(f"Stage 2B mAP@0.5: {best_map50:.4f}")
        else:
            results["stage2b"] = {"note": "mAP column not found in results.csv"}
    else:
        log.warning("Stage 2B results.csv not found (not trained yet)")
        results["stage2b"] = {"error": "not trained yet"}

    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved to: {out_path}")
    return results


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

def cli_main():
    """Entry point for the `geo-intel-pipeline` console script (pyproject.toml)."""
    import sys as _sys
    _run_cli(_sys.argv[1:])


def _run_cli(argv=None):
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
    # Honour the argv parameter so cli_main(sys.argv[1:]) works as advertised.
    args = ap.parse_args(argv)

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
        # Use a real error instead of `assert` (assertions are stripped under
        # `python -O`, which would silently produce a NoneType error later).
        if not args.tif:
            ap.error("--tif is required for --mode infer")
        infer(str(args.tif), str(args.out))
    elif mode == "all":
        preprocess(data_root)
        train_all()
        evaluate()


if __name__ == "__main__":
    _run_cli()
