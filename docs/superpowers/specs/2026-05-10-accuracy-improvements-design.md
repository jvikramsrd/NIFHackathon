# Accuracy Improvements Design — 2026-05-10

## Goal
Improve end-to-end prediction accuracy across all three pipeline stages (segmentation, rooftop classification, infrastructure detection) through inference-side fixes and training config corrections.

## Changes by File

### `inference/pipeline.py`

| Location | Change | Reason |
|---|---|---|
| `_to_uint8()` | Percentile clipping (2nd–98th) instead of min-max | Robust to outlier / saturated pixels in satellite imagery |
| `_segment()` line ~263 | Remove `min(..., 128)` overlap cap | Config specifies 192px; cap was silently degrading tile blending |
| Both `tta_predict()` calls | `fast_tta=False` (reads `CFG.FAST_TTA`) | Full 3-scale × 8-fold TTA (24 passes) vs 8; significant accuracy gain |
| `_classify_rooftops()` threshold | Per-class conf thresholds from `CFG.STAGE2A["stage2a_conf_thresh"]` | Blanket 0.55 cutoff conflates hard classes with easy ones |

### `config.py`

| Key | Old | New | Reason |
|---|---|---|---|
| `STAGE1["crf_iter"]` | 5 | 10 | 5 iterations insufficient for CRF convergence |
| `STAGE1["min_fg_ratio"]` | 0.003 | 0.01 | Near-empty patches add training noise |
| `STAGE2A["stage2a_conf_thresh"]` | (absent) | `{'RCC':0.45,'Tiled':0.55,'Tin':0.50,'Other':0.40}` | Per-class calibration |
| `STAGE2B["soft_nms_sigma"]` | 0.9 | 0.5 | 0.9 barely suppresses overlapping boxes |
| `STAGE2B["class_conf_thresh"]["well"]` | 0.03 | 0.10 | 3% threshold produces massive false-positive well detections |
| `FAST_TTA` | (absent) | `False` | Module-level toggle for TTA mode |

### `models/stage2_models.py`

| Location | Change | Reason |
|---|---|---|
| `RooftopClassifier.__init__` line ~114 | `m=cfg.get("arcface_m", 0.50)`, `s=cfg.get("arcface_s", 30.0)` | Config sets `arcface_m=0.55` but it was ignored; hardcoded 0.50 used instead |

## Architecture (unchanged)
- Stage 1: UNet++ / MiT-B5 / scSE — no change
- Stage 2A: ConvNeXt-Large / ArcFace — no change
- Stage 2B: YOLOv9-OBB / SAHI — no change

## Expected Impact
- **_to_uint8 fix**: affects all three stages; prevents feature suppression on high-dynamic-range imagery
- **TTA upgrade**: ~1–3 mIoU pts on Stage 1; marginal time cost acceptable for final inference
- **Overlap cap removal**: reduces seam artifacts at tile boundaries
- **CRF 5→10**: sharper building/road boundaries, fewer misclassified edge pixels
- **ArcFace margin fix**: uses the tuned 0.55 value which provides tighter class separation
- **min_fg_ratio 0.003→0.01**: cleaner training data; fewer wasted gradient steps on background tiles
- **Soft-NMS sigma 0.9→0.5**: removes infrastructure duplicate detections
- **Well threshold 0.03→0.10**: reduces false-positive wells substantially
