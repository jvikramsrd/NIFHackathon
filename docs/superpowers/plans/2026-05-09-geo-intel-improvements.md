# Geo-Intel Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all training blockers in run_pipeline.py, apply accuracy improvements across all 3 stages, and build a PyQt6 desktop GUI.

**Architecture:** Approach C — fix broken code first to unblock training, then apply targeted config/code accuracy changes, then build the GUI as a self-contained layer on top. Each task is independently verifiable.

**Tech Stack:** PyTorch, segmentation-models-pytorch, ultralytics, SAHI, albumentations, PyQt6, matplotlib, rasterio, fiona

---

## File Map

| File | Action | Reason |
|------|--------|--------|
| `run_pipeline.py` | Modify | Remove duplicate block, implement `evaluate()`, fix `train_all()` |
| `config.py` | Modify | CPU fallback, road weight 3.5→4.5, arcface_m 0.50→0.55, SAHI overlap 0.30→0.40, well thresh lower by 0.05, add `agnostic_nms` |
| `data/dataset.py` | Modify | Add `RandomFog` + `RandomSunFlare` to Stage 2A augmentation pipeline |
| `models/stage2_models.py` | Modify | Wire `sahi_overlap_ratio` and `agnostic_nms` from config instead of hardcoded values |
| `requirements.txt` | Modify | Add `PyQt6` |
| `gui.py` | Create | PyQt6 desktop app with Pipeline Runner, Map Viewer, Results tabs |

---

## Task 1: Fix run_pipeline.py — remove duplicate block and fix train_all()

**Files:**
- Modify: `run_pipeline.py:37-169`

- [ ] **Step 1: Confirm the duplicate block**

Run:
```
python -c "import ast, sys; ast.parse(open('run_pipeline.py').read()); print('syntax ok')"
```
Expected: `SyntaxError` because `import config as CFG` appears at module level inside an incomplete function body (line 55), confirming the corruption.

- [ ] **Step 2: Rewrite run_pipeline.py with the duplicate removed and train_all() fixed**

Replace the entire file content with:

```python
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

    _header("STAGE 1 — Semantic Segmentation  (UNet++ mit_b5)")
    train_stage1()
    clear_cuda_cache()

    _header("STAGE 2A — Rooftop Classifier  (ConvNeXt-Large)")
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
```

- [ ] **Step 3: Verify syntax is clean**

Run:
```
python -c "import ast; ast.parse(open('run_pipeline.py').read()); print('syntax ok')"
```
Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add run_pipeline.py
git commit -m "fix: remove duplicate block, implement evaluate(), fix train_all()"
```

---

## Task 2: Config — CPU fallback + all accuracy improvements

**Files:**
- Modify: `config.py:52` (DEVICE)
- Modify: `config.py:172` (class_weights road 3.5→4.5)
- Modify: `config.py:213` (arcface_m 0.50→0.55)
- Modify: `config.py:263` (sahi_overlap_ratio 0.30→0.40)
- Modify: `config.py:268` (well conf_thresh 0.08→0.03)
- Modify: `config.py` STAGE2B dict (add agnostic_nms)

- [ ] **Step 1: Write a test to pin the config values**

Create `tests/test_config_values.py`:

```python
def test_device_is_torch_device():
    import torch
    import config as CFG
    assert isinstance(CFG.DEVICE, torch.device)

def test_road_class_weight_is_4_5():
    import config as CFG
    # index 2 is road
    assert CFG.STAGE1["class_weights"][2] == 4.5

def test_arcface_m_is_0_55():
    import config as CFG
    assert CFG.STAGE2A["arcface_m"] == 0.55

def test_sahi_overlap_is_0_40():
    import config as CFG
    assert CFG.STAGE2B["sahi_overlap_ratio"] == 0.40

def test_well_conf_thresh_is_0_03():
    import config as CFG
    assert CFG.STAGE2B["class_conf_thresh"]["well"] == 0.03

def test_agnostic_nms_enabled():
    import config as CFG
    assert CFG.STAGE2B.get("agnostic_nms") is True
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```
python -m pytest tests/test_config_values.py -v
```
Expected: All 6 tests FAIL

- [ ] **Step 3: Apply all config changes**

In `config.py`, make the following 6 changes:

**Change 1** — line 52, DEVICE CPU fallback:
```python
# Before:
DEVICE = "cuda"

# After:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Change 2** — line 172, road class weight:
```python
# Before:
    class_weights=[0.30, 1.80, 3.50, 2.20],

# After:
    class_weights=[0.30, 1.80, 4.50, 2.20],
```

**Change 3** — line 213, ArcFace margin:
```python
# Before:
    arcface_m=0.50,

# After:
    arcface_m=0.55,
```

**Change 4** — line 263, SAHI overlap:
```python
# Before:
    sahi_overlap_ratio=0.30,

# After:
    sahi_overlap_ratio=0.40,
```

**Change 5** — well confidence threshold (line 268):
```python
# Before:
        'well': 0.08,

# After:
        'well': 0.03,
```

**Change 6** — add agnostic_nms to STAGE2B dict (add after `soft_nms_sigma` line):
```python
    soft_nms_sigma=0.9,
    agnostic_nms=True,
    use_sahi=True,
```

- [ ] **Step 4: Run tests — expect all pass**

Run:
```
python -m pytest tests/test_config_values.py -v
```
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_values.py
git commit -m "feat: accuracy improvements — road weight 4.5x, arcface_m 0.55, SAHI overlap 0.40, well thresh 0.03, agnostic_nms"
```

---

## Task 3: Stage 2A augmentations — add RandomFog and RandomSunFlare

**Files:**
- Modify: `data/dataset.py:243-279` (get_clf_train_transforms)

- [ ] **Step 1: Write a test to confirm the new augmentations are in the pipeline**

Add to `tests/test_core_components.py`:

```python
def test_clf_train_transforms_include_fog_and_sunflare():
    from data.dataset import get_clf_train_transforms
    import albumentations as A

    tf = get_clf_train_transforms(224)
    type_names = [type(t).__name__ for t in tf.transforms]

    assert "RandomFog" in type_names, "RandomFog missing from clf train transforms"
    assert "RandomSunFlare" in type_names, "RandomSunFlare missing from clf train transforms"
```

- [ ] **Step 2: Run test to confirm it fails**

Run:
```
python -m pytest tests/test_core_components.py::test_clf_train_transforms_include_fog_and_sunflare -v
```
Expected: FAIL — `AssertionError: RandomFog missing`

- [ ] **Step 3: Add RandomFog and RandomSunFlare to get_clf_train_transforms in data/dataset.py**

In `get_clf_train_transforms`, after the `A.RandomShadow` line (currently line 272), add:

```python
    # Simulate haze / low-visibility conditions common in SVAMITVA imagery
    A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.10),
    # Simulate direct sun glare on metallic rooftops (Tin / RCC)
    A.RandomSunFlare(
        src_radius=80,
        num_flare_circles_lower=2,
        num_flare_circles_upper=6,
        p=0.15,
    ),
```

The final `transforms.extend([...])` block in `get_clf_train_transforms` should look like:

```python
    transforms.extend(
        [
            A.ColorJitter(0.35, 0.35, 0.25, 0.08, p=0.7),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.16)),
                    A.GaussianBlur(blur_limit=3),
                    A.Sharpen(alpha=(0.2, 0.5)),
                ],
                p=0.35,
            ),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.10),
            A.RandomSunFlare(
                src_radius=80,
                num_flare_circles_lower=2,
                num_flare_circles_upper=6,
                p=0.15,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
```

- [ ] **Step 4: Run test — expect pass**

Run:
```
python -m pytest tests/test_core_components.py::test_clf_train_transforms_include_fog_and_sunflare -v
```
Expected: PASS

- [ ] **Step 5: Run all tests to check for regressions**

Run:
```
python -m pytest tests/ -v
```
Expected: All tests PASS (the existing 4 + the new augmentation test)

- [ ] **Step 6: Commit**

```bash
git add data/dataset.py tests/test_core_components.py
git commit -m "feat: add RandomFog and RandomSunFlare to Stage 2A rooftop augmentation pipeline"
```

---

## Task 4: Wire SAHI overlap and agnostic_nms from config in stage2_models.py

**Files:**
- Modify: `models/stage2_models.py:358-397` (InfrastructureDetector.predict)

- [ ] **Step 1: Write a failing test**

Add to `tests/test_core_components.py`:

```python
def test_sahi_overlap_reads_from_config():
    """InfrastructureDetector.predict must use sahi_overlap_ratio from cfg, not a hardcoded 0.30."""
    import inspect
    from models.stage2_models import InfrastructureDetector
    src = inspect.getsource(InfrastructureDetector.predict)
    # Must NOT contain the hardcoded 0.30 literal for overlap
    assert "overlap_height_ratio=0.30" not in src, \
        "overlap_height_ratio is hardcoded 0.30 — should read from cfg['sahi_overlap_ratio']"
```

- [ ] **Step 2: Run test to confirm it fails**

Run:
```
python -m pytest tests/test_core_components.py::test_sahi_overlap_reads_from_config -v
```
Expected: FAIL

- [ ] **Step 3: Update InfrastructureDetector.predict in models/stage2_models.py**

In the `predict` method, replace the hardcoded SAHI call (around line 358–369) with:

```python
        if use_sahi:
            sahi_model = self._get_sahi_model()
            if sahi_model is not None:
                try:
                    from sahi.predict import get_sliced_prediction  # type: ignore
                    overlap_ratio = float(self.cfg.get("sahi_overlap_ratio", 0.30))
                    slice_size = int(self.cfg.get("sahi_slice_size", 640))
                    result = get_sliced_prediction(
                        img_path,
                        sahi_model,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                        perform_standard_pred=True,
                        postprocess_type="NMM",
                        postprocess_match_threshold=0.50,
                        verbose=0,
                    )
```

Also update the standard YOLO fallback predict call (around line 391) to include `agnostic_nms`:

```python
        if not use_sahi or not raw_dets:
            results = self.model(
                img_path,
                conf=min(class_thresholds.values()) if class_thresholds else default_thresh,
                iou=self.cfg["iou_thresh"],
                max_det=self.cfg.get("max_det", 300),
                augment=True,
                agnostic_nms=bool(self.cfg.get("agnostic_nms", False)),
            )
```

- [ ] **Step 4: Run test — expect pass**

Run:
```
python -m pytest tests/test_core_components.py::test_sahi_overlap_reads_from_config -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run:
```
python -m pytest tests/ -v
```
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add models/stage2_models.py tests/test_core_components.py
git commit -m "feat: wire sahi_overlap_ratio and agnostic_nms from config in InfrastructureDetector"
```

---

## Task 5: Update requirements.txt with PyQt6

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add PyQt6 to requirements.txt**

Add to the `# ── Hardware / System` section:

```
# ── GUI ───────────────────────────────────────────────────────────────────────
PyQt6>=6.6.0
```

- [ ] **Step 2: Install PyQt6**

Run:
```
pip install PyQt6
```
Expected: Successfully installed PyQt6 (or already satisfied)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add PyQt6 for desktop GUI"
```

---

## Task 6: Build PyQt6 desktop GUI (gui.py)

**Files:**
- Create: `gui.py`

- [ ] **Step 1: Create gui.py with the full 3-tab app**

```python
"""
gui.py — Geo-Intel Pipeline Desktop App
Run with: python gui.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────


class PipelineTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["preprocess", "train_stage1", "train_stage2", "train_all", "evaluate", "infer"]
        )
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Data root picker
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data Root:"))
        self.data_label = QLabel(str(ROOT / "dataset"))
        self.data_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        data_row.addWidget(self.data_label, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._pick_data_root)
        data_row.addWidget(browse_btn)
        layout.addLayout(data_row)

        # Run / Stop row
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Live log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_view, stretch=1)

        # QProcess for subprocess
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

    def _pick_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Root", str(ROOT))
        if d:
            self.data_label.setText(d)

    def _run(self):
        mode = self.mode_combo.currentText()
        data_root = self.data_label.text()
        self.log_view.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start(
            sys.executable,
            [str(ROOT / "run_pipeline.py"), "--mode", mode, "--data_root", data_root],
        )

    def _stop(self):
        self.process.kill()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_view.appendPlainText("\n[Stopped by user]")

    def _on_stdout(self):
        text = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._append_log(text)

    def _on_stderr(self):
        text = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_view.appendPlainText(text.rstrip())
        for line in text.splitlines():
            if "Ep " in line and "/" in line:
                try:
                    part = line.split("Ep ")[1].split()[0]
                    curr, total = part.split("/")
                    self.progress.setValue(int(int(curr) / int(total) * 100))
                except Exception:
                    pass
            elif "epoch" in line.lower():
                try:
                    import re
                    m = re.search(r"(\d+)/(\d+)", line)
                    if m:
                        self.progress.setValue(int(int(m.group(1)) / int(m.group(2)) * 100))
                except Exception:
                    pass

    def _on_finished(self, exit_code: int, _exit_status):
        self.log_view.appendPlainText(f"\n[Process finished — exit code {exit_code}]")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if exit_code == 0 else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Map / Image Viewer
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # 0: background — black
        [255, 60, 60],    # 1: building — red
        [160, 160, 160],  # 2: road — gray
        [60, 100, 255],   # 3: waterbody — blue
    ],
    dtype=np.uint8,
)


class MapViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        open_btn = QPushButton("Open TIF…")
        open_btn.clicked.connect(self._open_tif)
        toolbar.addWidget(open_btn)

        load_mask_btn = QPushButton("Load Mask…")
        load_mask_btn.clicked.connect(self._load_mask)
        toolbar.addWidget(load_mask_btn)

        toolbar.addWidget(QLabel("Overlay opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(140)
        self.opacity_slider.valueChanged.connect(self._update_display)
        toolbar.addWidget(self.opacity_slider)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Side-by-side image panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_label = QLabel("Open a TIF to begin")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_label.setStyleSheet("background: #111; color: #aaa;")
        self.left_label.setMinimumSize(300, 300)

        self.right_label = QLabel("Load a prediction mask to see overlay")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_label.setStyleSheet("background: #111; color: #aaa;")
        self.right_label.setMinimumSize(300, 300)

        splitter.addWidget(self.left_label)
        splitter.addWidget(self.right_label)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # Legend
        legend_row = QHBoxLayout()
        for color_hex, name in [("#3c3c3c", "Background"), ("#ff3c3c", "Building"),
                                 ("#a0a0a0", "Road"), ("#3c64ff", "Waterbody")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color_hex}; font-size: 18px;")
            legend_row.addWidget(dot)
            legend_row.addWidget(QLabel(name))
            legend_row.addSpacing(12)
        legend_row.addStretch()
        layout.addLayout(legend_row)

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    def _open_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GeoTIFF", str(ROOT / "dataset"), "GeoTIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            import rasterio
            with rasterio.open(path) as src:
                bands = min(src.count, 3)
                data = src.read(list(range(1, bands + 1)))
            rgb = np.transpose(data, (1, 2, 0))
            if rgb.dtype != np.uint8:
                rgb = self._to_uint8(rgb)
            if rgb.shape[2] < 3:
                rgb = np.stack([rgb[:, :, 0]] * 3, axis=-1)
            self._rgb = rgb
            self._update_display()
        except Exception as e:
            self.left_label.setText(f"Error: {e}")

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation Mask", str(ROOT / "outputs"), "Image (*.png *.tif *.tiff)"
        )
        if not path:
            return
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                import rasterio
                with rasterio.open(path) as src:
                    mask = src.read(1)
            self._mask = mask
            self._update_display()
        except Exception as e:
            self.right_label.setText(f"Error: {e}")

    def _update_display(self):
        if self._rgb is None:
            return
        # Downscale for display (max 1024px on longest side)
        rgb_disp = self._fit(self._rgb, 1024)
        self._set_pixmap(self.left_label, rgb_disp)

        if self._mask is not None:
            mask_disp = cv2.resize(
                self._mask, (rgb_disp.shape[1], rgb_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = _CLASS_COLORS[np.clip(mask_disp, 0, len(_CLASS_COLORS) - 1)]
            alpha = self.opacity_slider.value() / 100.0
            blended = (
                rgb_disp.astype(np.float32) * (1 - alpha)
                + overlay.astype(np.float32) * alpha
            ).clip(0, 255).astype(np.uint8)
            self._set_pixmap(self.right_label, blended)
        else:
            self._set_pixmap(self.right_label, rgb_disp)

    @staticmethod
    def _fit(img: np.ndarray, max_px: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(max_px / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(min(arr.shape[2], 3)):
            ch = arr[:, :, i].astype(np.float32)
            mn, mx = ch.min(), ch.max()
            out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
        return out.astype(np.uint8)

    @staticmethod
    def _set_pixmap(label: QLabel, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        label.setPixmap(
            QPixmap.fromImage(img).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self._load_results)
        browse_btn = QPushButton("Browse Outputs…")
        browse_btn.clicked.connect(lambda: os.startfile(str(ROOT / "outputs")))
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(browse_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Stage", "mIoU", "Dice / F1", "Accuracy", "mAP@0.5"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        fig = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(fig)
        self.ax = fig.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

        self._load_results()

    def _load_results(self):
        results_path = ROOT / "outputs" / "results.json"
        if not results_path.exists():
            self.table.setRowCount(0)
            return

        with open(results_path) as f:
            results = json.load(f)

        self.table.setRowCount(0)
        chart_labels: list = []
        chart_values: list = []

        for stage_key, data in results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            def _cell(v) -> str:
                return f"{v:.4f}" if isinstance(v, float) else str(v)

            self.table.setItem(row, 0, QTableWidgetItem(stage_key))
            self.table.setItem(row, 1, QTableWidgetItem(_cell(data.get("mIoU", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(_cell(data.get("dice", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(_cell(data.get("accuracy", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(_cell(data.get("mAP_50", ""))))

            # Pick the primary metric for the bar chart
            primary = data.get("mIoU") or data.get("accuracy") or data.get("mAP_50")
            if isinstance(primary, float):
                chart_labels.append(stage_key)
                chart_values.append(primary)

        self.ax.clear()
        if chart_labels:
            bars = self.ax.bar(chart_labels, chart_values, color=["#4e79a7", "#f28e2b", "#59a14f"])
            self.ax.set_ylim(0, 1)
            self.ax.set_ylabel("Score")
            self.ax.set_title("Primary Metric per Stage")
            for bar, val in zip(bars, chart_values):
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geo-Intel Pipeline")
        self.resize(1100, 780)

        tabs = QTabWidget()
        tabs.addTab(PipelineTab(), "Pipeline Runner")
        tabs.addTab(MapViewerTab(), "Map Viewer")
        tabs.addTab(ResultsTab(), "Results")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify gui.py syntax**

Run:
```
python -c "import ast; ast.parse(open('gui.py').read()); print('syntax ok')"
```
Expected: `syntax ok`

- [ ] **Step 3: Verify imports work (PyQt6 + matplotlib)**

Run:
```
python -c "from PyQt6.QtWidgets import QApplication; from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg; print('imports ok')"
```
Expected: `imports ok`

- [ ] **Step 4: Commit**

```bash
git add gui.py
git commit -m "feat: add PyQt6 desktop GUI with Pipeline Runner, Map Viewer, and Results tabs"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Bug fix — duplicate block removed: Task 1 ✓
  - Bug fix — evaluate() implemented: Task 1 ✓
  - Bug fix — train_all() fixed: Task 1 ✓
  - CPU fallback: Task 2 ✓
  - Road weight 4.5×: Task 2 ✓
  - ArcFace margin 0.55: Task 2 ✓
  - SAHI overlap 0.40: Tasks 2 + 4 ✓
  - Well conf thresh lower by 0.05: Task 2 ✓
  - Agnostic NMS: Tasks 2 + 4 ✓
  - RandomFog + RandomSunFlare for Stage 2A: Task 3 ✓
  - PyQt6 GUI with all 3 tabs: Task 6 ✓
  - requirements.txt updated: Task 5 ✓

- [x] **No placeholders:** All steps have complete code.

- [x] **Type consistency:** `evaluate()` uses `SegmentationMetrics`, `split_dataset`, `split_clf_dataset` — all consistent with their definitions in utils/metrics.py and data/dataset.py.

- [x] **Scope:** 6 tasks, each independently committable and testable.
