# Geo-Intel Pipeline — Improvements Design Spec
**Date:** 2026-05-09  
**Goal:** Maximize hackathon score across all 3 pipeline stages, fix all training blockers, and add a PyQt6 desktop GUI.

---

## Context

The NIFHackathon Geo-Intel pipeline is a 3-stage deep learning system for drone orthophoto analysis:
- **Stage 1:** UNet++ semantic segmentation (buildings, roads, waterbodies)
- **Stage 2A:** ConvNeXt rooftop material classification (RCC / Tiled / Tin / Other)
- **Stage 2B:** YOLOv9 infrastructure detection (transformer / overhead tank / well)

Starting from scratch — no trained checkpoints. All 3 stages are evaluated with multiple metrics (mIoU, classification accuracy, mAP). No hard submission constraints.

---

## Section 1: Bug Fixes

These are hard blockers preventing end-to-end training from running.

### 1.1 `run_pipeline.py` — Duplicate Code Block
Lines ~55–73 repeat the preprocess imports and function header already defined at lines ~20–50, resulting from a merge/paste error. Remove the duplicate block entirely.

### 1.2 `run_pipeline.py` — `evaluate()` Undefined
`evaluate()` is called in `train_all()` and as a CLI mode but is never defined in the file. Implement it to:
1. Load trained checkpoints for each stage
2. Run inference on the validation split
3. Compute metrics via `utils/metrics.py` (mIoU/Dice for Stage 1, accuracy for Stage 2A, mAP@0.5 for Stage 2B)
4. Write results to `outputs/results.json`

### 1.3 `run_pipeline.py` — `train_all()` Undefined Variables
`train_all()` references `cfg`, `module`, and `val_ds` which are never defined in scope. Wire these to:
- `cfg` → the `Config` object from `config.py`
- `module` → the appropriate stage module import
- `val_ds` → the validation dataset constructed from `data/dataset.py`

### 1.4 Config — CPU Fallback
Add `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` auto-detection so the pipeline can run on CPU for testing without crashing.

---

## Section 2: Accuracy Improvements

Highest-ROI changes per stage targeting the evaluation metrics.

### 2.1 Stage 1 — Segmentation (mIoU)

**Class weight adjustment:**  
Increase road class weight from 3.5× → 4.5× in `config.py`. Roads are thin linear features frequently missed; higher weight penalises road omissions more strongly during training.

**LR Scheduler:**  
Replace flat LR decay with `CosineAnnealingWarmRestarts` (T_0=10, T_mult=2). This works better with SAM optimizer by allowing periodic re-exploration of the loss landscape.

**SWA validation:**  
Add per-epoch validation during the SWA averaging window to detect SWA collapse (where averaged weights degrade rather than improve) and roll back if mIoU drops >2 points.

### 2.2 Stage 2A — Rooftop Classification (Accuracy)

**Harder augmentations:**  
Add to the training augmentation pipeline in `data/dataset.py`:
- `RandomSunFlare` (p=0.15) — drone imagery shot at different times of day
- `RandomFog` (p=0.10) — haze common in SVAMITVA imagery
- `RandomShadow` (p=0.20) — building shadows create false material cues

**ArcFace margin:**  
Increase `m` from 0.50 → 0.55 in `config.py` (`STAGE2A.arcface_m`). Tighter angular margin improves RCC vs Tiled discrimination, the hardest pair in the dataset.

### 2.3 Stage 2B — Infrastructure Detection (mAP)

**SAHI overlap:**  
Increase `overlap_ratio` from 0.30 → 0.40 in `config.py`. Wider overlap ensures small objects (wells ~40px) at slice boundaries appear in at least one full slice context.

**Well confidence threshold:**  
Lower the per-class confidence threshold for `well` by 0.05 (e.g. 0.25 → 0.20). Wells are the smallest and most under-detected class; a lower threshold recovers true positives at acceptable precision cost.

**Agnostic NMS:**  
Enable `agnostic_nms: True` in YOLOv9 inference config to suppress duplicate cross-class detections near transformers (which are often co-located with overhead tanks).

---

## Section 3: Desktop GUI (PyQt6)

**Framework:** PyQt6. Handles image rendering, live log streaming, and embedded matplotlib charts better than Tkinter. Single dependency: `pip install PyQt6`.

**Entry point:** `gui.py` in the project root, launched via `python gui.py`.

### Tab 1: Pipeline Runner
- Dropdown: select mode (`Preprocess / Train Stage 1 / Train Stage 2 / Full Pipeline / Inference`)
- Folder pickers: "Data Root" and "Output Folder"
- `Run` button: launches the selected `run_pipeline.py` mode as a `QProcess` subprocess
- Live scrolling log: stdout/stderr streamed line-by-line into a `QPlainTextEdit`
- Progress bar: driven by log-line parsing (e.g. `Epoch 12/150` → 8%)
- `Stop` button: kills the subprocess cleanly via `QProcess.kill()`

### Tab 2: Map / Image Viewer
- Left panel: file browser (`QTreeView`) scoped to the dataset directory, showing `.tif` files
- Right panel: side-by-side display — raw RGB tile (left) vs. predicted segmentation overlay (right), colour-coded by class (building=red, road=yellow, water=blue)
- Overlay opacity slider: `QSlider` 0–100%, blends mask over RGB in real-time
- "Load Shapefile" button: overlays exported polygon/point shapefiles rendered as outlines on the image using Fiona + OpenCV

### Tab 3: Results Dashboard
- Metrics table (`QTableWidget`): per-run rows with columns for mIoU, Dice, per-class IoU (Stage 1), Accuracy (Stage 2A), mAP@0.5 (Stage 2B)
- Populated by parsing `outputs/results.json`, refreshed on tab focus
- Bar chart: `matplotlib FigureCanvasQTAgg` embedded below the table, showing metric comparison across runs
- "Browse Outputs" button: opens `outputs/` in Windows Explorer via `os.startfile()`

---

## Implementation Order

1. Fix `run_pipeline.py` (Section 1) — unblocks everything
2. Apply accuracy improvements (Section 2) — config and code changes only, no new dependencies
3. Build `gui.py` (Section 3) — self-contained, depends on the fixed pipeline

---

## Dependencies Added

| Package | Purpose |
|---------|---------|
| `PyQt6` | Desktop GUI framework |
| `matplotlib` | Embedded bar charts in Results Dashboard |

Both are `pip install`-only; no system packages required.
