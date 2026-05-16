# GeoIntel Pipeline — Complete Project Reference

> This document covers every component of the codebase: purpose, architecture, data flow, all files, every improvement made across the full git history, and current configuration values. Read this to understand the project from scratch.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Dataset — SVAMITVA](#2-dataset--svamitva)
3. [Directory Layout](#3-directory-layout)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Stage 1 — Semantic Segmentation](#5-stage-1--semantic-segmentation)
6. [Stage 2A — Rooftop Material Classification](#6-stage-2a--rooftop-material-classification)
7. [Stage 2B — Infrastructure Detection](#7-stage-2b--infrastructure-detection)
8. [Data Preprocessing Pipeline](#8-data-preprocessing-pipeline)
9. [Training Pipeline](#9-training-pipeline)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Post-Processing](#11-post-processing)
12. [Utilities Reference](#12-utilities-reference)
13. [Entry Points](#13-entry-points)
14. [GUI — Operator Console](#14-gui--operator-console)
15. [Configuration Reference](#15-configuration-reference)
16. [Complete History of Improvements](#16-complete-history-of-improvements)

---

## 1. What This Project Does

This is an **AI-powered geospatial analysis pipeline** built for the **SVAMITVA (Survey of Villages and Mapping with Improvised Technology in Village Areas)** scheme — India's drone-based village land mapping programme.

Given a high-resolution drone orthophoto (GeoTIFF), the pipeline:

1. **Segments** the image into buildings, roads, waterbodies, and background.
2. **Classifies** each building's rooftop material (RCC, Tiled, Tin, Other).
3. **Detects** small infrastructure objects (electric transformers, overhead water tanks, hand pumps/wells).

Outputs are GIS-ready vector shapefiles (`.shp`) and GeoPackages (`.gpkg`) — ready to open in QGIS or ArcGIS.

**Target hardware:** Windows 11, RTX A4000 16 GB, i9-13900K, 32 GB RAM.

---

## 2. Dataset — SVAMITVA

### Folder structure (two sub-datasets)
```
dataset/
  cg/   ← Chhattisgarh villages (5 TIF orthos + 1 ECW)
  pb/   ← Punjab villages (5 TIF orthos + 2 ECW)
```

Each folder contains:
- One or more large GeoTIFF orthorasters (some exceed 70 GB uncompressed)
- ECW duplicates of TIFs → **automatically skipped** (TIF preferred)
- `.tif.pyrx` pyramid files → **skipped**
- ArcGIS `.lock` files → **skipped**
- Shared shapefiles that annotate **all** rasters in the folder:

| Shapefile | Role | Config key |
|---|---|---|
| `Built_Up_Area_type.shp` / `Built_Up_Area_typ.shp` | Buildings + roof type | class_id=1 |
| `Road.shp` / `Road_Centre_Line.shp` | Roads | class_id=2 |
| `Water_Body.shp` / `Water_Body_Line.shp` | Waterbodies | class_id=3 |
| `Utility.shp` / `Utility_Poly.shp` | Infrastructure points/polygons | Stage 2B |
| `Bridge.shp` / `Railway.shp` | Treated as road class | class_id=2 |

### Segmentation classes
| ID | Name | Color |
|---|---|---|
| 0 | Background | Black |
| 1 | Building | Red |
| 2 | Road | Grey |
| 3 | Waterbody | Blue |

### Rooftop material classes (Stage 2A)
`RCC`, `Tiled`, `Tin`, `Other`

Map from raw DBF values (e.g. `Pucca_RCC` → `RCC`, `Pucca_Tiled` → `Tiled`) is in `config.ROOF_TYPE_MAP`.

### Infrastructure classes (Stage 2B)
`transformer`, `overhead_tank`, `well`

Map from raw DBF values (e.g. `electric_transformer` → `transformer`) is in `config.INFRA_TYPE_MAP`.

---

## 3. Directory Layout

```
NIFHackathon/
├── config.py                   ← All hyperparameters and paths
├── gui.py                      ← PyQt6 desktop operator console
├── infer_folder.py             ← Batch inference entry point
├── run_stage2b.py              ← Stage 2B standalone training runner
├── export_models.py            ← ONNX / TorchScript export
│
├── data/
│   ├── dataset.py              ← Dataset classes + augmentation pipelines
│   └── preprocessing.py        ← Raw TIF → patches/crops/YOLO labels
│
├── models/
│   ├── stage1_segmentation.py  ← Unet/MiT-B4 model, losses, TTA
│   └── stage2_models.py        ← ConvNeXt+ArcFace classifier, YOLO detector
│
├── inference/
│   └── pipeline.py             ← Full 3-stage inference orchestration
│
├── train/
│   ├── train_stage1.py         ← Stage 1 training loop
│   ├── train_stage2.py         ← Stage 2A + 2B training loops
│   └── launch_ddp.py           ← Multi-GPU DDP launcher
│
├── utils/
│   ├── checkpointing.py        ← Atomic checkpoint save
│   ├── ddp.py                  ← Distributed training helpers
│   ├── hardware.py             ← Device setup, EMA, compile wrappers
│   ├── logger.py               ← Structured logging + crash reporter
│   ├── metrics.py              ← IoU, F1, confusion matrix
│   ├── postprocess.py          ← CRF, morphology, vectorization, SHPs
│   ├── sam.py                  ← Sharpness-Aware Minimization optimizer
│   └── window.py               ← Cosine blending window for tile overlap
│
├── dataset/                    ← Raw SVAMITVA data (not in git)
│   ├── patches/                ← 512px segmentation training tiles
│   ├── patch_masks/            ← Corresponding class masks
│   ├── building_crops/         ← Per-class rooftop image crops
│   ├── yolo_infra/             ← YOLO-format infrastructure tiles
│   └── masks/                  ← Full-raster segmentation masks
│
├── checkpoints/                ← Saved model weights (not in git)
│   ├── stage1_best.pth
│   ├── stage1_last.pth
│   ├── stage1_swa.pth
│   ├── stage2a_best.pth
│   └── stage2b_yolov9e/weights/best.pt
│
└── outputs/vectorized/         ← Pipeline outputs (shapefiles, gpkg)
```

---

## 4. High-Level Architecture

```
GeoTIFF ortho
      │
      ▼
[_to_uint8]  ← percentile normalisation (2nd–98th per channel)
      │
      ├─────────────────────────────────────────────────────────┐
      ▼                                                         │
[STAGE 1]  Unet / MiT-B4                                        │
  Tiled inference (512px patches, 192px overlap)                │
  3-scale × 8-fold TTA (24 passes, controlled by FAST_TTA)      │
  Dense CRF refinement (10 iterations)                          │
  Morphological cleanup + watershed building separation         │
  ↓                                                             │
  Segmentation mask (H×W uint8, 4 classes)                      │
  Vectorised → building.shp, road.shp, waterbody.shp            │
      │                                                         │
      ▼                                                         │
[STAGE 2A]  ConvNeXt-Large / ArcFace                            │
  Constrained to building polygons from Stage 1                 │
  Per-building bbox crop (224×224, 15% padding)                 │
  3-scale × 8-fold TTA (24 passes)                              │
  Per-class confidence thresholds                               │
  ↓                                                             │
  Roof material per building → building_rooftop.shp             │
      │                                                         │
      ▼                                                         │
[STAGE 2B]  YOLOv9-OBB / SAHI                                   │
  Context-gated tiling (skip tiles far from buildings/roads)    │
  1280px tiles, 512px overlap                                    │
  SAHI sliced inference (640px slices, 40% overlap)             │
  Per-class confidence thresholds                               │
  Soft-NMS Gaussian (σ=0.5) deduplication                       │
  ↓                                                             │
  Infrastructure detections → infrastructure.shp                │
      └───────────── all_features.gpkg ───────────────────────┘
```

---

## 5. Stage 1 — Semantic Segmentation

### Model: `models/stage1_segmentation.py`

**Architecture:** Unet with:
- Encoder: **MiT-B4** (Mix Transformer) — pretrained on ImageNet
- Decoder: **scSE attention** (channel + spatial squeeze-excitation) on every decoder block
- Activation: `None` (raw logits, softmax applied at loss/inference time)

**Why Unet:** It is the most stable SMP decoder with MiT encoders, trains faster than UNet++/MAnet variants, and keeps clean skip-feature detail for building, road, and water boundaries.

**Why MiT-B4:** Hierarchical Vision Transformer. It captures fine-grained texture and settlement context while cutting VRAM and iteration time versus MiT-B5.

### Loss: `TriLoss`

A weighted sum of six losses:

| Loss | Weight | Purpose |
|---|---|---|
| Cosine-log Dice | 0.40 | Directly optimises overlap ratio; smoother gradients than standard Dice near 0/1 |
| Cross-Entropy (label smooth 0.05) | 0.15 | Stable pixel-level supervision |
| Focal (γ=2.0) | 0.15 | Down-weights easy background pixels, focuses on hard boundaries |
| Boundary / Hausdorff | 0.15 | Penalises distance between predicted and GT edges |
| Lovász-Softmax | 0.15 | Directly optimises mIoU (the actual evaluation metric) |
| Instance-touching separation | 0.10 | Penalises high building probability at inter-building gaps; separates touching buildings |

**Class weights:** `[0.30, 1.80, 4.50, 2.20]` for background/building/road/waterbody.  
Road gets the highest weight (4.5×) because it is narrow and easily confused with shadows.

**Auxiliary-output ready:** The loss still supports auxiliary logits if a future architecture returns them, but the production Unet path uses a single main head.

### TTA: `tta_predict`

**Full TTA** (default, `FAST_TTA=False`):
- 3 scales: 0.875× (wider context), 1.0× (base), 1.25× (fine detail)
- 8 folds per scale: 4 rotations × 2 flip states (D4 symmetry group)
- Total: **24 forward passes** per batch
- Scale weights: 0.875→0.8, 1.0→1.0, 1.25→0.9

**Fast TTA** (`FAST_TTA=True`):
- 2 scales: 1.0×, 1.25×
- 4 folds per scale
- Total: **8 forward passes** — ~3× faster, lower accuracy

### Tiled inference

- Patch size: **512px**
- Overlap: **192px** (was incorrectly capped at 128px — now uses the configured value)
- Blending: **spline window** (power=2 cosine taper) — smooth cross-fade at tile boundaries, no seams
- Batch size: 16 tiles per GPU forward pass

### Dense CRF refinement

Applied after tiled inference to sharpen boundaries. Uses `pydensecrf`:
- Pairwise Gaussian (position): `sxy=3.0`, weight=3.0
- Pairwise Bilateral (position + colour): `sxy=80`, `srgb=13`, weight=10
- Expert per-class compatibility matrix: building↔waterbody confusion penalised 3×, building↔road penalised 1.5×
- Texture-aware bilateral: Sobel gradient magnitude modulates the bilateral weight
- **Iterations: 10** (was 5 — insufficient for convergence)
- Tiled to avoid OOM on large orthos (2048px tiles, 256px overlap)

### Morphological cleanup

After CRF, applied per class:
- **Buildings:** close (7px ellipse) → remove blobs < 80px → **watershed separation** of touching buildings
- **Roads:** close (18px ellipse, 2 iterations) → open (3px ellipse) → remove small blobs
- **Waterbodies:** close (11px ellipse) → remove blobs < 160px

**Watershed building separation:**  
Distance transform → `peak_local_max` (min 10px between seeds) → marker-controlled `watershed`. Inserts 1-px gap between adjacent building instances before vectorisation.

---

## 6. Stage 2A — Rooftop Material Classification

### Model: `models/stage2_models.py → RooftopClassifier`

**Architecture:** ConvNeXt-Large backbone (timm) with:
- Global average pooling (`num_classes=0`)
- Feature projection trunk: `LayerNorm → Dropout(0.5) → Linear(1536→768) → GELU → LayerNorm → Dropout(0.3)`
- **ArcFace head** (Deng et al., 2019): angular margin loss for tighter inter-class separation
  - `s=30.0`, `m=0.55` (margin; was hardcoded to 0.50, ignoring config — now reads from config)
  - Pushes class embeddings further apart in cosine space, reducing RCC↔Tiled confusion

**Why ConvNeXt-Large:** Better than ViT-based models for fine-grained texture classification at small crop sizes (224px). ConvNeXt is optimised for `channels_last` memory layout → 15–25% throughput gain on Ampere.

**Why ArcFace:** Rooftop materials (RCC vs Tiled) are visually similar. Standard softmax produces overlapping class clusters in embedding space. ArcFace enforces an angular margin that makes decision boundaries crisper without needing more training data.

### TTA at inference

3-scale × 8-fold = 24 passes (same D4 group as Stage 1):
- Scale 0.875×: pad then resize → shows wider context around the building
- Scale 1.0×: base crop
- Scale 1.25×: center-crop then resize → zooms into fine texture (RCC grit, tile ridges)

### Per-class confidence thresholds

If the model's max-probability falls below the threshold for the predicted class, the prediction is overridden to "Other":

| Class | Threshold | Reasoning |
|---|---|---|
| RCC | 0.45 | Most visually distinctive — concrete slab is unambiguous |
| Tiled | 0.55 | Commonly confused with Tin at glancing angles |
| Tin | 0.50 | Moderate — metallic sheen is usually clear |
| Other | 0.40 | Catch-all; being permissive here reduces missed buildings |

*(Was a single blanket 0.55 for all classes — now per-class)*

### Inference flow

1. Load `building.shp` produced by Stage 1
2. For each polygon: compute bbox in pixel space, add 15% padding, crop from ortho
3. Skip polygons smaller than `min_crop_px=40` px
4. Resize crop to 224×224, apply val transforms
5. Batch in groups of 64, run through classifier with 24-fold TTA
6. Apply per-class threshold
7. Write results back to `building_rooftop.shp`

---

## 7. Stage 2B — Infrastructure Detection

### Model: `models/stage2_models.py → InfrastructureDetector`

**Backend:** **YOLOv9e-OBB** (Oriented Bounding Box variant) via `ultralytics`.
Falls back to standard YOLOv9e if OBB weights unavailable.
Falls back to Faster R-CNN (torchvision) if ultralytics not installed.

**Why OBB:** Infrastructure objects (transformers on poles, tanks) have irregular orientations. OBB preserves orientation, avoids the large enclosing-rectangle problem of axis-aligned boxes.

**SAHI (Slicing Aided Hyper Inference):**
- Input tiles are 1280px; SAHI additionally slices into 640px sub-tiles with 40% overlap
- Sub-tile results merged with `NMM` (Non-Maximum Merging, threshold 0.50)
- Also runs standard full-tile prediction alongside sliced prediction
- Critical for detecting small objects (well pumps, ~1m diameter at 5cm/px GSD)

**Per-class confidence thresholds:**

| Class | Threshold | Notes |
|---|---|---|
| transformer | 0.20 | Electric transformers are large and distinctive |
| overhead_tank | 0.12 | Tanks vary in size; moderate threshold |
| well | 0.10 | Very small objects; was 0.03 (extreme false positive rate) |

**Soft-NMS Gaussian (σ=0.5):**  
Instead of hard-suppressing overlapping boxes, decays their scores: `score *= exp(-IoU²/σ)`.  
Preserves two legitimate transformers on adjacent poles (which hard NMS would drop).  
σ=0.5 is the paper's recommended value. Was σ=0.9 (barely any suppression).

**Context-gated tiling:**  
Before running detection on a 1280px tile, checks if that tile intersects any building or road polygon (with a 64px buffer). Tiles with no nearby structures are skipped entirely. On village orthos this skips 40–70% of tiles.

---

## 8. Data Preprocessing Pipeline

### Entry: `data/preprocessing.py → preprocess_folder()`

Handles the entire raw-data-to-training-data conversion. Key design:

**Memory-safe strip processing:**  
Giant TIFs (up to 213,734 × 112,836 px = ~72 GB RGB) are never loaded in full. Instead:
1. Read 4096-row horizontal strip RGB (~2.6 GB)
2. Burn SHP mask for that strip (~0.9 GB)
3. Tile the strip → save patches to disk
4. Discard both arrays (GC)
5. Advance to next strip

Peak RAM per raster: **~3.5 GB** (safe on 32 GB).

**STRtree spatial index:**  
SHP geometries are loaded once per raster into a `shapely.STRtree`. Per strip, only geometries that intersect the strip bounding box are rasterised — no brute-force iteration.

**Parallelism:**  
Multiple rasters processed in parallel via `ProcessPoolExecutor` (up to 5 workers, ~17 GB RAM total). Within each raster, tile writes use `ThreadPoolExecutor` (P-core count − 2).

**Patch filtering:**  
Tiles where foreground pixels / total pixels < `min_fg_ratio` (currently **0.01**) are dropped. This prevents training on near-empty background patches.

**Building crops (Stage 2A):**  
Uses `rasterio.windows.Window` to read only the building bbox from the raster — never loads the full strip for crop extraction. Uses `cv2.INTER_AREA` (correct for downscaling) instead of `INTER_LINEAR`.

**YOLO label generation (Stage 2B):**  
Object-centered tile strategy: each tile is centered on an infrastructure object cluster rather than grid-snapped. Class-specific bounding box sizes: transformer=100px, tank=80px, well=40px. Negative tile sampling (30% of positive count) teaches YOLO what non-infrastructure looks like.

**Normalisation (`_to_uint8`):**  
Preprocessing's `_to_uint8` also uses percentile stretch (2nd–98th, excluding zero pixels). Zero pixels are excluded from percentile computation to avoid no-data areas compressing the valid range.

---

## 9. Training Pipeline

### Stage 1: `train/train_stage1.py`

**Optimiser:** **SAM (Sharpness-Aware Minimization)** wrapping AdamW.  
SAM finds flatter loss minima, which generalise better. It does two forward+backward passes per step. With Unet/MiT-B4, the code keeps the encoder fixed and only reduces batch size on low-VRAM GPUs.

**Layer-wise learning rates:**
- Encoder (MiT-B4): `lr × 0.1 = 2e-5` — fine-tunes pretrained features gently
- Decoder: `lr = 2e-4` — trains from scratch aggressively

**Scheduler:** OneCycleLR (`pct_start=0.1`, `div_factor=25`, `final_div_factor=1e4`)

**SWA (Stochastic Weight Averaging):**  
Starts at epoch 75% of total. Maintains a running average of model weights. At the end of training, SWA BN statistics are updated with one pass over the training data. SWA checkpoint saved separately as `stage1_swa.pth`. SWA typically gives +0.5–1.5 mIoU over the best single checkpoint.

**EMA (Exponential Moving Average):**  
`decay=0.9998`. EMA shadow is applied for validation each epoch (then restored for training). Best checkpoint saves EMA weights, not raw weights.

**Multi-scale training:**  
50% of batches are randomly resized to one of `(0.5×, 1.0×, 1.5×)` then resized back to 512px. Builds scale invariance without storing multiple patch sizes.

**CutMix augmentation:**  
20% probability per batch. Applied at the image+mask level (mask is cut simultaneously with the image region).

**Data augmentation (Albumentations):**
- Multi-scale crop (256/512/768px → resize to 512px)
- D4 symmetry (H-flip, V-flip, rot90, transpose)
- Random brightness/contrast, HSV jitter, CLAHE, gamma
- Random fog (simulate haze): 15% probability
- Gaussian noise, blur, motion blur, median blur: 35%
- Elastic transform, perspective, grid distortion: 40%
- Coarse dropout (tree canopy occlusion): 35%
- ImageNet normalisation (mean/std)

**Stratified train/val split:**  
Patches are sorted by foreground ratio into 4 quartile strata, then each stratum is split 85/15. This ensures the validation set reflects the full difficulty distribution (not just easy low-foreground tiles).

**Early stopping:** patience=18 epochs. Decision broadcast from rank-0 in DDP mode.

**DDP support:** Fully implemented via `utils/ddp.py`. Launch with `train/launch_ddp.py`. Single-GPU falls through to standard training with no code changes.

**Gradient checkpointing:** Enabled on MiT-B4 encoder if supported (`model.encoder.set_grad_checkpointing(True)`).

**Gradient clipping:** `clip_grad_norm_(1.0)`. Skips optimizer step if norm > 10.0 (spike guard).

### Stage 2A: `train/train_stage2.py`

**Optimiser:** SAM + AdamW, `lr=5e-5`.  
**Augmentation:** RandAugment (n=2, m=7) + ColorJitter + shadow + fog + sun flare.  
**MixUp + CutMix:** Applied during training to reduce overfitting on small rooftop datasets.  
**ArcFace training:** `forward_train()` passes labels to the ArcFace head so the angular margin is applied during training; `forward()` (inference) passes `labels=None` → returns scaled cosine logits.

### Stage 2B: `run_stage2b.py`

Wraps `InfrastructureDetector.train()` which calls `YOLO.train()` with all config parameters. Supports OBB mode. Uses YOLOv9e-OBB with aggressive copy-paste (0.30), mosaic (1.0), and HSV augmentation.

---

## 10. Inference Pipeline

### Entry: `inference/pipeline.py → GeoIntelPipeline`

**Constructor:** Loads all three models once. Applies `channels_last` memory format to ConvNeXt (15–25% NHWC throughput gain on Ampere). torch.compile disabled on Windows (no Triton).

**`run(tif_path, out_dir)`** orchestrates the full pipeline:

```
1. Open TIF → read RGB bands → _to_uint8 (percentile normalise)
2. Stage 1: _segment() → prob_map (C, H, W) float32
3. Optional CRF refinement on prob_map
4. argmax → seg_mask (H, W) uint8
5. Morphological cleanup → seg_mask
6. Save seg_mask → {prefix}_segmask.tif
7. Vectorise → {prefix}_building.shp, _road.shp, _waterbody.shp, _all_features.gpkg
8. Stage 2A: _classify_rooftops() using building.shp
   → merge labels → {prefix}_building_rooftop.shp
9. Stage 2B: _gather_context_polygons() → _detect()
   → {prefix}_infrastructure.shp
```

**Fallback on file lock:** If `_segmask.tif` can't be written (e.g. open in QGIS), writes to a timestamped fallback path rather than crashing.

**Batch inference:** `infer_folder.py` loads models once, then runs `.run()` on every `.tif`/`.tiff` in a directory tree.

---

## 11. Post-Processing

### File: `utils/postprocess.py`

**`apply_dense_crf()`:** Tiled CRF (2048px tiles, 256px overlap, cosine blending). Runs serially (one tile at a time) to avoid multiprocessing issues on Windows with pydensecrf.

**`clean_segmentation_mask()`:** Per-class morphological pipeline + watershed building separation (detailed in Stage 1 section).

**`mask_to_shapefile()`:** Rasterises each class layer, fixes topology with `shapely.make_valid`, simplifies polygons (`tolerance=0.5`), explodes multi-part features, filters by minimum area per class (building: 80px, road: 120px, waterbody: 160px). Saves per-class SHP and combined GeoPackage.

**`merge_rooftop_labels()`:** Joins Stage 2A predictions back to the building GeoDataFrame as a `roof_pred` attribute column.

**`detections_to_shapefile()`:** OBB detections → rotated rectangle polygons in geo-space. Standard-box detections → centroid points. Preserves the orientation angle from the YOLO OBB output.

**`clean_vector_geometries()`:** Fixes invalid geometries, simplifies, explodes multipart, filters by area, removes invalid results.

---

## 12. Utilities Reference

### `utils/hardware.py`

- `setup()`: Configures CUDA, TF32, cuDNN benchmark, Flash Attention (SDPA), OMP/MKL thread counts for i9-13900K (8 P-cores pinned).
- `EMA`: Exponential Moving Average. `apply_shadow()` swaps model to EMA weights; `restore()` swaps back.
- `compile_model()`: Wraps `torch.compile` with try/except. Disabled on Windows.
- `to_channels_last()`: Converts a module to NHWC memory layout (for ConvNeXt).
- `cl_input()`: Converts an input tensor to `channels_last` contiguous.
- `get_amp_context()`: Returns appropriate `torch.amp.autocast` context for the dtype (bfloat16/float16/float32).
- `vram_stats()`: Returns GPU name + allocated/reserved VRAM string.

### `utils/sam.py`

**SAM optimizer** (Sharpness-Aware Minimisation, Foret et al. 2021). Wraps any base optimizer.

- `first_step()`: Perturbs weights in gradient direction (finds sharper neighborhood).
- `second_step()`: Computes gradient at perturbed point, takes actual optimizer step.
- Fixed for PyTorch 2.x: uses `param_groups[i]["defaults"]` pattern (replaced old `_defaults` access).

### `utils/ddp.py`

Distributed Data Parallel helpers:
- `setup_ddp()`: Initialises `dist.init_process_group` if `WORLD_SIZE > 1`.
- `wrap_ddp()`: Wraps module in `DistributedDataParallel` if enabled.
- `make_loader()`: Adds `DistributedSampler` automatically when DDP is active.
- `is_main_process()`: True only on rank 0 (controls logging/checkpointing).
- `set_epoch()`: Calls `sampler.set_epoch()` for proper per-epoch shuffle in DDP.
- `cleanup_ddp()`: Calls `dist.destroy_process_group()`.

### `utils/checkpointing.py`

**`atomic_torch_save()`:** Saves to a `.tmp` file first, then `os.replace()`. Prevents corrupt checkpoints on crash mid-write.

### `utils/metrics.py`

**`SegmentationMetrics`:** Accumulates confusion matrix, computes per-class IoU, F1, mean IoU, pixel accuracy.  
**`ClassificationMetrics`:** Accuracy, macro F1, per-class precision/recall/F1.

### `utils/logger.py`

- `get_logger(name)`: Returns a configured logger with timestamps.
- `crash_logged(log, context)`: Context manager; catches all exceptions, logs traceback + context, re-raises.

### `utils/window.py`

**`cosine_window(size, overlap, power)`:** Shared cosine taper utility used by both `_segment()` (Stage 1 tile blending) and `apply_dense_crf()` (CRF tile blending). Power=2 gives smooth C¹-continuous taper.

---

## 13. Entry Points

| Script | Purpose | Usage |
|---|---|---|
| `inference/pipeline.py` | Single-image inference | `python inference/pipeline.py --tif path.tif --out ./out` |
| `infer_folder.py` | Batch inference on a directory | `python infer_folder.py --test_folder ./test_images --out_folder ./results` |
| `train/train_stage1.py` | Train Stage 1 segmentation | `python train/train_stage1.py` |
| `train/train_stage2.py` | Train Stage 2A classifier | `python train/train_stage2.py` |
| `run_stage2b.py` | Train Stage 2B YOLO detector | `python run_stage2b.py` |
| `train/launch_ddp.py` | Multi-GPU training launcher | `torchrun --nproc_per_node=N train/launch_ddp.py` |
| `export_models.py` | Export to ONNX / TorchScript | `python export_models.py` |
| `gui.py` | Desktop operator console | `python gui.py` |

---

## 14. GUI — Operator Console

**File:** `gui.py`  
**Framework:** PyQt6 + Matplotlib  
**Theme:** Custom Geo-Intel dark/light palette with animated elements

### Tabs

**Pipeline Runner tab:**
- Checkpoint file pickers (Stage 1/2A/2B)
- Input TIF picker
- Output directory picker
- Run button → launches `inference/pipeline.py` as a `QProcess` subprocess
- Live log output with ANSI colour stripping and scrolling
- Progress bar driven by log line patterns

**Map Viewer tab:**
- Opens the output segmentation mask TIF
- Displays with colour-coded overlay (building=red, road=grey, water=blue)
- OpenCV-based rendering in a Matplotlib canvas
- Zoom/pan controls

**Results tab:**
- Shapefile statistics table (feature counts, class distributions)
- Reads the `_all_features.gpkg` GeoPackage from the output directory

The GUI calls `infer_folder.py` or `inference/pipeline.py` as a subprocess, so it does not need GPU resources itself and can run on any machine.

---

## 15. Configuration Reference

All configuration lives in `config.py`. Below are the current values with explanations.

### Paths
```python
ROOT        = Path(__file__).parent
DATA_ROOT   = ROOT / "dataset"
PATCH_DIR   = DATA_ROOT / "patches"          # 512px seg training tiles
MASK_DIR    = DATA_ROOT / "patch_masks"       # class masks
CROP_DIR    = DATA_ROOT / "building_crops"    # Stage 2A crops
YOLO_DIR    = DATA_ROOT / "yolo_infra"        # Stage 2B tiles
CKPT_DIR    = ROOT / "checkpoints"
LOG_DIR     = ROOT / "logs"
OUT_DIR     = ROOT / "outputs/vectorized"
TRAIN_MASKS = DATA_ROOT / "masks"
```

### Hardware
```python
DEVICE         = cuda / mps / cpu   # auto-detected
AMP_DTYPE      = bfloat16           # bf16 on CUDA/ROCm Ampere+, fp16 on MPS
COMPILE_ENABLED= False              # torch.compile disabled on Windows (no Triton)
COMPILE_MODE   = "reduce-overhead"
FAST_TTA       = False              # True=8 passes, False=24 passes (more accurate)
NUM_WORKERS    = 10
PIN_MEMORY     = True
PREFETCH_FACTOR= 3
```

### Stage 1 — Segmentation
```python
num_classes        = 4
class_names        = ['background', 'building', 'road', 'waterbody']
arch               = 'Unet'
encoder            = 'mit_b4'
encoder_weights    = 'imagenet'
patch_size         = 512
patch_sizes        = (512,)
overlap            = 128
batch_size         = 8                 # auto-reduced on low-VRAM GPUs
grad_accum         = 4
lr                 = 2e-4
encoder_lr_mult    = 0.1               # encoder LR = lr * 0.1
weight_decay       = 1e-4
epochs             = 80
warmup_epochs      = 3
# Loss weights
dice_weight        = 0.40
bce_weight         = 0.15
focal_weight       = 0.15
boundary_weight    = 0.15
lovasz_weight      = 0.15
touching_weight    = 0.10
focal_gamma        = 2.0
class_weights      = [0.30, 1.80, 4.50, 2.20]
label_smoothing    = 0.05
# Training tricks
use_sam            = True
sam_rho            = 0.05
sam_adaptive       = True
ms_training        = True
ms_scales          = (0.5, 1.0, 1.5)
use_swa            = True
swa_lr             = 2e-5
swa_start_frac     = 0.75
use_ema            = True
ema_decay          = 0.9998
cutmix_alpha       = 1.0
drop_path_rate     = 0.2
val_fraction       = 0.15
# Inference
crf_inference      = True
crf_iter           = 10               # iterations (was 5)
# Data filtering
min_fg_ratio       = 0.01             # minimum foreground fraction per patch (was 0.003)
neg_tile_ratio     = 0.15
# Vectorisation
min_building_area_px          = 80
polygon_min_area_px           = {building:80, road:120, waterbody:160}
polygon_simplify_tolerance    = 0.5
```

### Stage 2A — Rooftop Classification
```python
num_classes        = 4
class_names        = ['RCC', 'Tiled', 'Tin', 'Other']
arch               = 'convnext_large'
crop_size          = 224
min_crop_px        = 40
batch_size         = 32
lr                 = 5e-5
epochs             = 150
tta_steps          = 24
use_arcface        = True
arcface_s          = 30.0
arcface_m          = 0.55            # now correctly read from config (was hardcoded 0.50)
drop_path_rate     = 0.4
use_ema            = True
ema_decay          = 0.9995
stage2a_conf_thresh= {RCC:0.45, Tiled:0.55, Tin:0.50, Other:0.40}  # per-class
```

### Stage 2B — Infrastructure Detection
```python
class_names        = ['transformer', 'overhead_tank', 'well']
model_variant      = 'yolov9e'
use_obb            = True
obb_model_variant  = 'yolov9e-obb'
img_size           = 1280
epochs             = 200
conf_thresh        = 0.10
iou_thresh         = 0.60
max_det            = 1000
overlap            = 512
class_buffer_px    = {transformer:100, overhead_tank:80, well:40}
soft_nms_sigma     = 0.5             # Gaussian decay bandwidth (was 0.9)
agnostic_nms       = True
use_sahi           = True
sahi_slice_size    = 640
sahi_overlap_ratio = 0.40
class_conf_thresh  = {transformer:0.20, overhead_tank:0.12, well:0.10}  # well was 0.03
neg_tile_ratio     = 0.3
```

---

## 16. Complete History of Improvements

All improvements made to this codebase, in chronological order from oldest to newest.

---

### Initial commit — Baseline pipeline

- Basic inference pipeline: Stage 1 (UNet), Stage 2A (classifier), Stage 2B (YOLO)
- Simple min-max normalisation in `_to_uint8`
- Basic augmentation (flip, rotate, colour jitter)
- Standard AdamW optimiser, no SAM, no EMA
- Single-GPU only

---

### Early iterative improvements

- Added DVC pipeline configuration (`dvc.yaml`, `params.yaml`) for experiment tracking
- Added `activate.bat` for Windows venv activation
- Multiple rounds of code cleanup and config tuning

---

### feat: SAM + DDP + reproducible installer (`fb618f4`)

**What was added:**
- **SAM optimizer** (`utils/sam.py`): Sharpness-Aware Minimisation wrapping AdamW. Two forward+backward passes per step → flatter loss minima → better generalisation.
- **DDP support** (`utils/ddp.py`): Multi-GPU training via `DistributedDataParallel`. Single-GPU falls through transparently.
- **EMA** (`utils/hardware.py`): Exponential Moving Average over model weights. Validation uses EMA weights; best checkpoint saves EMA, not raw weights.
- **SWA**: Stochastic Weight Averaging from epoch 75%, with BN stat update after training.
- **Modular training pipelines**: `train_stage1.py`, `train_stage2.py` as importable functions.
- **Cross-platform installer**: `install.sh` (macOS/Linux) + `setup_venv.bat` (Windows) detect the right PyTorch wheel (CUDA / ROCm / MPS / CPU) and provision a `venv`.
- **Atomic checkpointing** (`utils/checkpointing.py`): Writes to `.tmp` first, then `os.replace()`.
- _(Note: a prior Dockerfile was removed in v1.x — PyInstaller binaries are now the recommended distribution; see `geo_intel.spec` and `geo_intel_cli.spec`.)_

---

### fix: remove duplicate block, implement evaluate(), fix train_all() (`f00f318`)

- Removed accidentally duplicated code block in training loop
- Implemented `InfrastructureDetector.evaluate()` method
- Fixed `train_all()` orchestration function

---

### feat: accuracy improvements v0.1 (`f0bd3da`)

**Config changes:**
- Road class weight raised to **4.5×** (from lower value) — roads are narrow and underrepresented
- `arcface_m` set to **0.55** in config (was 0.50) — tighter angular margin between rooftop classes
- SAHI overlap ratio set to **0.40** (was 0.30) — more overlap catches small objects at tile boundaries
- Well confidence threshold set to 0.03 — intentionally aggressive for recall (later raised to 0.10)
- `agnostic_nms=True` — class-agnostic NMS prevents same-location duplicate detections of different classes

---

### feat: add RandomFog and RandomSunFlare to Stage 2A augmentation (`f0bd3da`)

- Added `A.RandomFog` (fog_coef 0.05–0.15, prob=0.10) to rooftop training augmentation
- Added `A.RandomSunFlare` (prob=0.15) to rooftop training augmentation
- Simulates real-world aerial imagery conditions (haze, lens flare from sun angles)
- Makes the rooftop classifier more robust to atmospheric scattering

---

### feat: wire sahi_overlap_ratio and agnostic_nms from config (`5a001e2`)

- `InfrastructureDetector.predict()` now reads `sahi_overlap_ratio` and `agnostic_nms` from `CFG.STAGE2B`
- Previously hardcoded; now tunable without touching model code

---

### feat: add PyQt6 desktop GUI (`34880c2`, `91aad98`)

- Full PyQt6 operator console (`gui.py`) with:
  - Pipeline Runner tab (QProcess subprocess launcher, live log output)
  - Map Viewer tab (OpenCV + Matplotlib segmentation overlay)
  - Results tab (shapefile statistics table)
- Custom Geo-Intel dark/light theme
- Animated progress indicators
- Decoupled from GPU — runs on any machine, calls inference scripts as subprocesses

---

### fix: SAM optimizer PyTorch 2.x compatibility (`9f4ddb9`)

- PyTorch 2.x renamed internal optimizer attribute from `_defaults` to `defaults`
- Fixed `utils/sam.py` to use `param_groups[i]["defaults"]` pattern
- Without this fix, SAM would throw `AttributeError` on PyTorch 2.x and silently fall back

---

### improvements v0.1 (`67e91b4`) — Major batch

This commit introduced a large set of improvements across the entire codebase:

**Stage 1 model:**
- Standardized Stage 1 on **Unet + MiT-B4** for production speed/accuracy balance
- Added **scSE decoder attention** on all decoder blocks
- Added **deep supervision** (auxiliary heads at stages 2/3/4, aux weights 0.4/0.2/0.1)
- Added **Lovász-Softmax loss** (directly optimises mIoU instead of a proxy)
- Added **instance-touching separation loss** (penalises merged adjacent buildings)
- Changed Dice to **cosine-log Dice** for smoother gradients near 0/1
- Upgraded TTA from 8-fold to **3-scale × 8-fold = 24 passes**
- Added **0.875× scale** to TTA (sees more context around large buildings)

**Stage 1 training:**
- Multi-scale training with crop sizes (256/512/768px → resize to 512px)
- Stratified train/val split by foreground ratio quartile
- OneCycleLR scheduler (replaced CosineAnnealingLR)
- VRAM auto-guard: auto-reduces batch size when SAM is active on ≤16 GB GPU
- VRAM auto-guard: keeps `mit_b4` fixed and reduces batch size on low-VRAM GPUs
- Gradient checkpointing on MiT-B4 encoder
- Gradient spike guard: skips optimizer step if grad norm > 10

**Stage 1 inference:**
- Spline window blending (was linear, now power-2 cosine taper → no seams)
- Shared `cosine_window` utility between `_segment()` and CRF tiling
- Dense CRF tiling with cosine blending (was full-image, OOM on large orthos)
- CRF: texture-aware bilateral using Sobel gradient magnitude
- CRF: per-class compatibility matrix (building↔waterbody 3×, building↔road 1.5×)
- Watershed building separation in morphological cleanup

**Stage 2A model:**
- Upgraded backbone from `convnext_base` → **`convnext_large`**
- Added **ArcFace head** (angular margin loss) replacing standard linear head
- Deeper feature projection trunk (LayerNorm + Dropout + GELU + LayerNorm)
- Drop path rate 0.4 (was 0.2) — stronger regularisation for larger backbone
- 3-scale TTA in classifier (was single-scale)
- Per-crop **instance normalisation** in `RooftopDataset` — equalises brightness across villages with different sun angles
- MixUp + CutMix augmentation in training

**Stage 2B model:**
- Added **SAHI** (Slicing Aided Hyper Inference) for small object recall
- Added **per-class confidence thresholds** (transformer/tank/well each tunable)
- Soft-NMS Gaussian sigma configurable (was hardcoded)
- OBB → rotated rectangle polygon export in `detections_to_shapefile()`
- Class-specific bounding box sizes in YOLO label generation

**Data preprocessing:**
- Strip-based processing for giant TIFs (never loads full raster)
- STRtree spatial index for SHP burning (vs brute-force iteration)
- ProcessPoolExecutor for parallel raster processing
- ThreadPoolExecutor for parallel tile writes
- Negative tile sampling for Stage 2B
- Object-centered tile strategy (vs grid-snapped)
- `_to_uint8` in preprocessing uses percentile stretch with zero-pixel exclusion

**Infrastructure:**
- `utils/window.py` extracted as shared module
- `utils/logger.py` structured logging + `crash_logged` context manager
- Atomic checkpoint save (`utils/checkpointing.py`)
- Multi-backend hardware support (CUDA / ROCm / MPS / CPU)
- Flash Attention via SDPA enabled

---

### Session improvements — 2026-05-10 (current session)

These are the improvements made in the most recent development session:

**`config.py`**
- `crf_iter`: **5 → 10** — CRF requires at least 10 iterations to converge on boundary refinement
- `min_fg_ratio`: **0.003 → 0.01** — patches with <0.3% foreground add noise without signal
- `soft_nms_sigma`: **0.9 → 0.5** — σ=0.9 barely penalises overlapping boxes; σ=0.5 is the paper's default
- `well` confidence threshold: **0.03 → 0.10** — 3% threshold was flooding output with false-positive wells
- Added `stage2a_conf_thresh = {RCC:0.45, Tiled:0.55, Tin:0.50, Other:0.40}` — per-class calibration
- Added `FAST_TTA = False` — module-level toggle for TTA mode (False = full 24-pass TTA)

**`inference/pipeline.py`**
- `_to_uint8()`: **min-max → 2nd–98th percentile clipping** — satellite imagery has frequent outliers (dead pixels, saturated areas). Min-max was compressing the entire valid range to 7/255 in tests; percentile gives 181/255. All downstream models receive proper input contrast.
- `_segment()` overlap: **removed `min(overlap, 128)` cap** — config specifies 192px; the cap was silently reducing overlap quality. Full 192px overlap is now used.
- Both `tta_predict()` calls: **`fast_tta=True` → `fast_tta=CFG.FAST_TTA`** — now controlled by config; defaults to full 24-pass TTA.
- `_classify_rooftops()` confidence gate: **hardcoded 0.55 → per-class thresholds from config** — reads `CFG.STAGE2A["stage2a_conf_thresh"]`; falls back to 0.55 if key missing.

**`models/stage2_models.py`**
- `ArcFaceHead` instantiation: **hardcoded `m=0.50`, `s=30.0` → `cfg.get("arcface_m", 0.50)`, `cfg.get("arcface_s", 30.0)`** — `config.py` had `arcface_m=0.55` for 11 commits but the value was never used. The tuned margin is now active.

---

*End of document.*
