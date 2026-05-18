# GeoIntel Pipeline — Complete Project Reference

> Every component, every knob, every "why." Read this end-to-end to understand
> the project from scratch. Values quoted below are the **live values** in
> `config.py` as of this document (cross-checked against the source — if the
> code ever drifts from this document, the code is authoritative).

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why a Three-Stage Pipeline](#2-why-a-three-stage-pipeline)
3. [Dataset — SVAMITVA](#3-dataset--svamitva)
4. [Directory Layout](#4-directory-layout)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Stage 1 — Semantic Segmentation](#6-stage-1--semantic-segmentation)
7. [Stage 2A — Rooftop Material Classification](#7-stage-2a--rooftop-material-classification)
8. [Stage 2B — Infrastructure Detection](#8-stage-2b--infrastructure-detection)
9. [Data Preprocessing](#9-data-preprocessing)
10. [Training Pipeline](#10-training-pipeline)
11. [Inference Pipeline](#11-inference-pipeline)
12. [Post-Processing](#12-post-processing)
13. [Utilities Reference](#13-utilities-reference)
14. [Entry Points](#14-entry-points)
15. [GUI — Operator Console](#15-gui--operator-console)
16. [Configuration Reference (live values + why)](#16-configuration-reference)
17. [A4000-Specific Optimization Notes](#17-a4000-specific-optimization-notes)
18. [Improvement History (recent → older)](#18-improvement-history)

---

## 1. What This Project Does

**GeoIntel** is an AI pipeline for the **SVAMITVA** scheme (Survey of Villages
and Mapping with Improvised Technology in Village Areas), India's drone-based
village land-mapping programme.

Given a high-resolution drone orthophoto (GeoTIFF or ECW) of a village, it
produces GIS-ready vector outputs:

1. **Segmentation** — pixel-level mask of buildings / roads / waterbodies / background.
2. **Rooftop classification** — material of every detected building (RCC / Tiled / Tin / Other).
3. **Infrastructure detection** — bounding boxes for small features (transformers, overhead water tanks, hand pumps / wells).

Outputs are `.shp` (per-class) and a combined `.gpkg` — load straight into QGIS or ArcGIS.

**Target hardware:** Windows 11, NVIDIA **RTX A4000 16 GB** (Ampere, CC 8.6),
i9-13900K, 32 GB RAM. The code also runs on AMD ROCm, Apple Silicon MPS, and CPU
with graceful AMP-dtype fallback (`bfloat16` → `float16` → `float32`).

---

## 2. Why a Three-Stage Pipeline

A single end-to-end network would have to learn three very different tasks at
three very different scales: pixel-accurate boundary regression (segmentation),
fine-grained texture classification (rooftop material), and small-object
detection (well pumps ≈ 15 px wide). Splitting the problem lets each stage use
the right architecture, the right input size, and the right loss:

| Stage | Task | Why a dedicated model |
|---|---|---|
| 1 | Pixel segmentation | Needs full-image context + dense per-pixel output → an encoder-decoder (MAnet/MiT-B4) at 512 px |
| 2A | Building-level classification | Needs only the cropped rooftop + tight angular margin → ConvNeXt-Large + ArcFace at 224 px |
| 2B | Small-object detection | Needs high-resolution sliding-window inference → YOLOv9e-OBB at 1280 px with SAHI |

**Stage 2A is *gated* by Stage 1's building polygons** — it never sees the full
ortho, only the 224 px crops Stage 1 identified as buildings. This is roughly
two orders of magnitude less compute than running the classifier on every
pixel, and it makes the "Other" class meaningful (it only fires on things we
already believe are buildings).

**Stage 2B is *gated* by Stage 1's context polygons** — tiles that don't
intersect a building/road/waterbody (with a 128 px buffer) are skipped entirely
before they reach YOLO. On rural orthos this skips 40–70% of tiles.

---

## 3. Dataset — SVAMITVA

### Folder layout (two sub-datasets)

```
dataset/
  cg/   ← Chhattisgarh villages (5 TIF orthos + 1 ECW)
  pb/   ← Punjab villages         (5 TIF orthos + 2 ECW)
```

Each sub-folder contains:
- One or more giant GeoTIFFs (up to ~72 GB uncompressed RGB)
- ECW duplicates of TIFs → **automatically skipped** (TIF preferred when both exist)
- `.tif.pyrx` pyramid files → **skipped** (they're index caches, not imagery)
- ArcGIS `.lock` files → **skipped**
- Shared shapefiles that annotate **all** rasters in the folder

### Shapefile → class mapping (`config.SHP_LAYER_ROLES`)

| Shapefile stem | Maps to | Attribute column |
|---|---|---|
| `Built_Up_Area_type` / `Built_Up_Area_typ` (PB-truncated) | class 1 (Building) | `type` |
| `Road` / `Road_Centre_Line` | class 2 (Road) | `road_type` |
| `Water_Body` / `Water_Body_Line` / `Waterbody_Point` | class 3 (Waterbody) | `water_type` |
| `Bridge` | class 2 (Road) | `bridge_type` |
| `Railway` | class 2 (Road) | `railway_type` |
| `Utility` / `Utility_Poly` / `Utility_Poly_` (PB) | (no seg class) — Stage 2B only | `utility_type` |

Bridges and railways are merged into "Road" because all three serve the same
role in connectivity-based downstream tasks, and their footprints are visually
indistinguishable from drone altitude.

### Segmentation classes (Stage 1)

| ID | Name | Render colour |
|---|---|---|
| 0 | background | black |
| 1 | building | red |
| 2 | road | grey |
| 3 | waterbody | blue |

### Rooftop classes (Stage 2A) — `config.ROOF_TYPE_MAP`

Raw SVAMITVA DBF values normalise to four canonical classes:

| Canonical | Raw value examples |
|---|---|
| `RCC` | `Pucca_RCC`, `Pucca_RCC_Slab`, `concrete`, `concrete_roof` |
| `Tiled` | `Pucca_Tiled`, `mangalore_tile`, `tile` |
| `Tin` | `Pucca_Tin`, `tin_roof`, `metal_roof`, `galvanized`, `pucca_asbestos` |
| `Other` | `semi_pucca`, `kuccha`, `other`, `others` |

Asbestos is mapped to `Tin` because both classes show identical metallic-sheen
textures at drone GSD and the downstream usefulness is the same (non-permanent
roof material).

### Infrastructure classes (Stage 2B) — `config.INFRA_TYPE_MAP`

| Canonical | Raw value examples |
|---|---|
| `transformer` | `electric_transformer`, `dt`, `distribution_transformer` |
| `overhead_tank` | `overhead_water_tank`, `OHT`, `OHSR`, `water_tank` |
| `well` | `hand_pump`, `well`, `tube_well`, `tubewell` |

---

## 4. Directory Layout

```
NIFHackathon/
├── config.py                   ← Single source of truth for hyperparameters & paths
├── gui.py                      ← PyQt6 desktop operator console
├── run_pipeline.py             ← Master CLI (preprocess / train / evaluate / infer / all)
├── infer_folder.py             ← Batch inference entry point
├── run_stage2b.py              ← Stage 2B standalone training runner
├── export_models.py            ← ONNX / TorchScript export
├── _setup_verify.py            ← Post-install smoke test
├── build.py                    ← PyInstaller binary builder
│
├── data/
│   ├── dataset.py              ← Dataset classes + Albumentations augmentation
│   └── preprocessing.py        ← Raw TIF → patches/crops/YOLO labels (strip-streamed)
│
├── models/
│   ├── stage1_segmentation.py  ← MAnet/MiT-B4 + TriLoss + TTA
│   └── stage2_models.py        ← ConvNeXt+ArcFace + YOLOv9 wrapper + Soft-NMS
│
├── inference/
│   └── pipeline.py             ← Full 3-stage inference orchestration
│
├── train/
│   ├── train_stage1.py         ← Stage 1 training loop (DDP-aware)
│   ├── train_stage2.py         ← Stage 2A + 2B training loops
│   └── launch_ddp.py           ← Multi-GPU torchrun launcher
│
├── utils/
│   ├── checkpointing.py        ← Atomic checkpoint save (.tmp + replace)
│   ├── ddp.py                  ← DistributedDataParallel helpers
│   ├── ecw_compat.py           ← ECW → GeoTIFF auto-conversion
│   ├── hardware.py             ← Device setup, EMA, channels_last, AMP, VRAM stats
│   ├── logger.py               ← Structured logging + crash_logged context manager
│   ├── metrics.py              ← Segmentation mIoU + detection mAP
│   ├── postprocess.py          ← CRF, morphology, vectorization, SHP writers
│   ├── sam.py                  ← Sharpness-Aware Minimisation (batched _foreach_*)
│   └── window.py               ← Cosine blending window for tile overlap
│
├── tests/
│   ├── test_config_values.py   ← Config invariants
│   ├── test_core_components.py ← Component-level unit tests
│   └── test_optimizations.py   ← Regression tests for vectorised paths
│
├── dataset/                    ← Raw SVAMITVA data (gitignored)
│   ├── patches/                ← 512 px segmentation training tiles
│   ├── patch_masks/            ← Corresponding class masks
│   ├── building_crops/         ← Per-class rooftop image crops
│   ├── yolo_infra/             ← YOLO-format infrastructure tiles
│   └── masks/                  ← Full-raster segmentation masks
│
├── checkpoints/                ← Saved model weights (gitignored)
│   ├── stage1_best.pth         ← EMA weights at best val mIoU
│   ├── stage1_last.pth         ← Resumable training state (model + opt + sched)
│   ├── stage1_swa.pth          ← SWA-averaged weights (after epoch 60)
│   ├── stage2a_best.pth
│   ├── stage2a_last.pth
│   └── stage2b_yolo11l/weights/best.pt
│
└── outputs/vectorized/         ← Inference outputs (.shp, .gpkg)
```

---

## 5. High-Level Architecture

```
GeoTIFF / ECW ortho (potentially 70 GB+ uncompressed)
      │
      ▼
[_to_uint8]   ← 2nd–98th percentile per-channel stretch, zero-pixel excluded
      │
      ▼
[STAGE 1]  MAnet decoder + MiT-B4 encoder, bf16 + channels_last
      • Tiled inference: 512 px patches, 128 px overlap (cosine blend)
      • TTA: 3-scale × D4 symmetry = 24 forward passes (or 8 via FAST_TTA)
      • DenseCRF: 10 iterations, per-class compatibility matrix
      • Morphological cleanup + watershed building separation
      ↓
      Segmentation mask (H × W uint8)
      Vectorised → {prefix}_building.shp, _road.shp, _waterbody.shp, _all_features.gpkg
      │
      ▼
[STAGE 2A]  ConvNeXt-Large + ArcFace head, bf16 + channels_last
      • Constrained to building polygons from Stage 1
      • 224 px crops with 15 % padding, INTER_LINEAR resize
      • Per-class confidence thresholds (RCC/Tiled/Tin/Other)
      • 8-fold TTA per scale × 3 scales = up to 24 passes
      ↓
      → {prefix}_building_rooftop.shp
      │
      ▼
[STAGE 2B]  YOLOv9e-OBB + SAHI sliced inference
      • Context-gated tiling (skip tiles >128 px from any building/road)
      • 1280 px tiles, 512 px overlap
      • SAHI slices: 512 px sub-tiles, 45 % overlap, NMM merge
      • Per-class confidence thresholds + Soft-NMS Gaussian (σ=0.40)
      ↓
      → {prefix}_infrastructure.shp
```

---

## 6. Stage 1 — Semantic Segmentation

### Architecture (`models/stage1_segmentation.py`)

**Decoder: `MAnet`** (Multi-scale Attention Network) from `segmentation-models-pytorch`.

*Why MAnet over Unet:* MAnet's Position-wise Attention Block + Multi-scale Feature
Aggregation block sharpens irregular boundaries. The SVAMITVA villages have
non-rectangular building outlines, variable-width dirt roads, and irregular lake
edges — exactly the case where a plain Unet decoder smears edges. MAnet
preserves them.

*Note on UnetPlusPlus:* `smp.UnetPlusPlus` has a hardcoded rejection of MiT
encoders (`decoders/unetplusplus/model.py`). If you ever want `UnetPlusPlus`
you must switch to `efficientnet-b4` or similar.

**Encoder: `mit_b4`** (Mix Transformer, ~62 M params), ImageNet-pretrained.

*Why MiT-B4 over MiT-B5 or a ResNet:* B4 is ~20 % faster than B5 and gives
nearly the same mIoU on this task. SegFormer-family transformer features
are strong for rooftop / road / water texture. ResNet encoders are faster but
lose 1–2 mIoU on the boundary classes.

**Decoder attention: `scse`** (concurrent Spatial + channel Squeeze-Excite) on every
decoder block.

*Why scSE:* free recalibration of skip-features at minimal compute cost.

**Channels-last layout:** the whole model is converted to NHWC before training
via `to_channels_last(module)`. On Ampere tensor cores, NHWC convolutions are
15–30 % faster than the default NCHW.

**Gradient checkpointing:** the training loop attempts
`module.model.encoder.set_grad_checkpointing(True)`. MiT encoders in `smp`
don't expose this hook, so it silently falls through. At 512 px in bf16, the
activation peak is ~10–11 GB at `batch_size=4`, which fits well inside the
A4000's 16 GB budget without checkpointing.

### Loss — `TriLoss`

Weighted sum of six terms (current config):

| Term | Weight | What it does | Why |
|---|---|---|---|
| Cosine-log Dice | 0.40 | `1 − (2·intersect+ε)/(union+ε)`, then `log(cosh(·))` | Directly targets overlap; `log(cosh)` smooths the gradient near 0 and 1 where Dice is normally degenerate |
| Cross-Entropy (label-smooth 0.08) | 0.15 | Standard per-pixel CE | Stable supervisory signal that anchors training when other losses are saturated |
| Focal (γ=2.0) | 0.15 | `(1−p_t)^γ · CE` per pixel | Down-weights easy background pixels — the bulk of every patch — so optimisation focuses on hard boundaries |
| Boundary / Hausdorff | 0.15 | Edge-map Dice + erosion-based Hausdorff approximation | Penalises distance between predicted and GT edges, not just overlap |
| Lovász-Softmax | 0.15 | Vectorised over classes via one sort + one cumsum | Directly optimises mIoU, which is what we report |
| Instance-touching separation | 0.10 | Penalises high building prob at inter-building gaps | Stops adjacent buildings from merging into one polygon |

**Class weights: `[0.20, 2.00, 5.00, 2.50]`** (background/building/road/waterbody)
and **label smoothing 0.08**.

*Why road = 5.0×:* roads are thin (≤ 3 m wide → 6 px at 50 cm GSD), so they
are massively under-represented in pixel-count vs buildings. Without
re-weighting, the model learns to predict "not road" everywhere and still
scores >95 % pixel accuracy. The 5× weight makes the loss notice.

*Why background = 0.2×:* background dominates pixel count; without
down-weighting the loss is mostly background CE.

### TTA — `tta_predict`

| Mode | Scales | Folds | Total forwards | Use case |
|---|---|---|---|---|
| Full (`FAST_TTA=False`, default) | 0.875 ×, 1.0 ×, 1.25 × | 8 (D4 symmetry: 4 rotations × 2 flip states) | **24** | Final inference, val every Nth epoch |
| Fast (`FAST_TTA=True`) | 1.0 ×, 1.25 × | 4 | **8** | Quick iteration / dev |

All augmented views for one scale are stacked into a single batched forward
(`mega = torch.cat(augs, dim=0)`) and chunked at `tta_chunk=256` for safety —
cutting kernel launches from N per scale to 1.

### Tiled inference

- Patch size **512**, overlap **128**, stride = 384.
- Blending: shared cosine window (`utils/window.cosine_window`, `power=2`) so
  there are no visible seams at tile boundaries.
- Batch size = 16 tiles per GPU forward.

### DenseCRF refinement

Applied to the prob-map after tiled inference, before argmax. Uses `pydensecrf`:

| Parameter | Value | Why |
|---|---|---|
| Pairwise Gaussian (position) | `sxy=3.0`, weight=3.0 | Local smoothness |
| Pairwise Bilateral (position + colour) | `sxy=80`, `srgb=13`, weight=10 | Long-range colour-aware smoothing |
| **Per-class compatibility matrix** | `building↔waterbody = 3×`, `building↔road = 1.5×`, diag=0 | The most damaging confusions get explicit extra penalty |
| Iterations | **10** | Krähenbühl & Koltun show 10 is needed for boundary convergence; 5 (older default) was insufficient |
| Tile size / overlap | 1024 / 128 | CRF is O(n²); smaller tiles are 4× faster with bounded bilateral kernels |
| Workers | `CFG.CRF_WORKERS = 4` | `ProcessPoolExecutor` parallelism over tiles |

### Morphological cleanup

Per class, before vectorisation:

- **Buildings (class 1):** close (7 px ellipse) → drop blobs < `min_building_area_px = 80 px` → **watershed instance separation** (distance transform → `peak_local_max` with min 10 px between seeds → marker-controlled `watershed`).
- **Roads (class 2):** close (`min_road_width_px + 15 = 18 px`, 2 iterations) → open (`min_road_width_px = 3 px`) → drop blobs < 40 px.
- **Waterbodies (class 3):** close (11 px) → drop blobs < 160 px.

Watershed separation is critical: without it, adjacent row-houses merge into
single polygons that the downstream rooftop classifier can't crop
individually.

### Vectorisation

`utils/postprocess.mask_to_shapefile` rasterises each class layer, fixes
topology with `shapely.make_valid` (falls back to `buffer(0)`), simplifies
with `polygon_simplify_tolerance = 0.5`, explodes multipart features, and
filters by per-class minimum area: `{building: 80, road: 120, waterbody: 160}`.

---

## 7. Stage 2A — Rooftop Material Classification

### Architecture (`models/stage2_models.py → RooftopClassifier`)

```
ConvNeXt-Large (timm, ImageNet pretrained, drop_path_rate=0.4)
  → Global Avg Pool (num_classes=0)
  → LayerNorm(1536)
  → Dropout(0.50)
  → Linear(1536 → 768)
  → GELU
  → LayerNorm(768)
  → Dropout(0.30)
  → ArcFaceHead(768 → 4, s=30.0, m=0.55)    [or nn.Linear if use_arcface=False]
```

**Why ConvNeXt-Large over ViT:** large-kernel depthwise convolutions in
ConvNeXt are the best fit for fine-grained texture classification at small
input sizes (224 px). ViTs need bigger inputs to be competitive on texture.
ConvNeXt is also one of the best `channels_last`-friendly architectures —
on Ampere it gains 15–25 %.

**Why ArcFace:** RCC and Tiled rooftops share visual statistics (both are
hard, often grey, broken into rectangular cells). Standard softmax produces
overlapping class clusters in embedding space. ArcFace enforces an additive
angular margin `m` between the target class and all others — decision
boundaries get crisp without needing more data. We use `s=30.0`, `m=0.55`.

**Why `m=0.55` not 0.50:** 0.55 gives the tightest margin we could push
without instability on this dataset. The original code hardcoded `m=0.50`
and ignored the config value for 11 commits; the bug was fixed and 0.55 is
now actually in effect.

**Why `drop_path_rate=0.4`:** ConvNeXt-Large is ~200 M params on a small
rooftop dataset. Aggressive stochastic depth is the main regulariser.

### Training loop highlights

- **Optimiser:** AdamW + SAM (`use_sam=True`, `rho=0.05`, `adaptive=True`).
- **Scheduler:** `SequentialLR(LinearLR warmup 10 % of epochs, CosineAnnealingWarmRestarts)`.
  Warm restarts are explicitly chosen over plain cosine to prevent
  catastrophic forgetting of high-confidence early decisions.
- **Augmentation:** RandAugment (n=2, m=7) + ColorJitter + MixUp (α=0.4) +
  CutMix (α=1.0). MixUp/CutMix picked randomly per batch.
- **Class weights:** `RooftopDataset.class_weights()` returns
  inverse-frequency normalised weights, computed via `np.bincount` (the
  earlier per-sample Python loop was the second-biggest startup cost).
- **EMA:** decay 0.9995. EMA shadow is applied for validation each epoch
  (then restored for training inside a try/finally). Best checkpoint saves EMA weights.
- **VRAM auto-guard for SAM:** if the GPU has ≤16.5 GB total VRAM and
  `batch_size > 8`, batch is halved (with `max(8, ...)` floor) at training
  start. Uses a *local* variable, not `cfg["batch_size"]`, so the global
  `CFG.STAGE2A` dict isn't mutated.
- **`MAX_STEPS_PER_EPOCH` cap:** mirrored from Stage 1 so a 30 GB dataset
  doesn't make individual epochs unbounded.
- **Gradient-norm spike skip:** if the pre-clip grad norm exceeds 10, the
  optimizer step is skipped (and the scheduler/EMA tick is also skipped via
  a `did_step` flag — otherwise a spike-skipped batch would silently consume
  a scheduler tick).

### Inference TTA — `RooftopClassifier.predict`

3 scales × up to 8 folds = up to 24 passes:

| Scale | Crop behaviour | Why |
|---|---|---|
| 0.875 × | Reflect-pad then resize to 224 × 224 | Shows wider context around the building (eaves, surroundings) |
| 1.0 × | Base crop | Reference view |
| 1.25 × | Centre-crop then resize | Zooms into fine texture (RCC grit, tile ridges) |

All folds for a scale are stacked into one forward; `tta_chunk=128` caps
images per kernel call.

### Per-class confidence thresholds

If the predicted-class probability is below its class threshold, the
prediction is overridden to **`Other`**:

| Class | Threshold | Reasoning |
|---|---|---|
| RCC | 0.45 | Most visually distinctive (slab) — confidence is reliable |
| Tiled | 0.55 | Often confused with Tin at glancing angles — be strict |
| Tin | 0.50 | Metallic sheen usually clear — moderate threshold |
| Other | 0.40 | Catch-all; permissive to reduce missed buildings |

### Inference flow

1. Load `{prefix}_building.shp` produced by Stage 1.
2. For each polygon: bbox in pixel space, add 15 % padding, crop from ortho.
3. Skip polygons smaller than `min_crop_px = 40 px`.
4. Resize crop to 224 × 224 with `INTER_LINEAR`, apply val transforms.
5. Batch in groups of 64; run with up to 24-fold TTA.
6. Apply per-class threshold; write `{prefix}_building_rooftop.shp`.

---

## 8. Stage 2B — Infrastructure Detection

### Backend (`models/stage2_models.py → InfrastructureDetector`)

**Primary:** `YOLOv11l` (axis-aligned) via `ultralytics`.
Falls back to torchvision `fasterrcnn_resnet50_fpn_v2` if `ultralytics` is
missing.

**Why YOLOv11l over YOLOv9e (previous choice):**
- ~2× faster at parity COCO accuracy (25 M params vs 58 M)
- C2PSA attention block gives a small but consistent recall lift on small
  objects (wells ~15 px, transformers ~30 px)
- Frees ~2.5 GB VRAM at imgsz=1280 → `batch_size` doubled from 2 to 4 on the A4000
- Actively maintained in `ultralytics`; YOLOv9-OBB was never a first-party
  release and we were silently falling back to AABB anyway

**Why AABB, not OBB:** the three target classes are rotationally symmetric
(circular wells, circular/square tanks) or near-square from a top-down drone
view. The angle parameter is mathematically undefined for circles and
4-way ambiguous for squares, so OBB regression on these objects is wasted
capacity at best and noisy at worst. AABB outputs Point centroids via
`detections_to_shapefile`, which is what GIS users actually want for an
infrastructure inventory layer. `cfg["obb_model_variant"]` is retained as a
no-op fallback in case OBB is ever re-enabled for a different class set.

### SAHI — Slicing Aided Hyper Inference

| Knob | Value | Why |
|---|---|---|
| Tile size (outer) | 1280 px | Matches YOLO training resolution |
| Tile overlap | 512 px | Big enough that any 100 px object sits fully inside at least one tile |
| Slice size (SAHI inner) | **512 px** | Smaller slices = better small-object recall at the cost of more forward passes |
| Slice overlap | **45 %** | High overlap so transformer-on-edge / well-on-edge is fully seen by at least one slice |
| Postprocess | `NMM` (Non-Maximum Merging, threshold 0.50) | Merges duplicates from overlapping slices instead of suppressing |
| Standard pred too | `True` | Runs full-tile pred *alongside* sliced pred; ensemble |

### Per-class confidence thresholds

| Class | Threshold | Why |
|---|---|---|
| transformer | 0.20 | Large, distinctive — confidence is high; threshold high to suppress lookalikes (boxes/junction-boxes) |
| overhead_tank | 0.12 | Tank sizes vary widely; moderate threshold |
| well | 0.10 | Tiny objects (~15–30 px); a too-strict threshold misses them all |

The minimum of these is what's actually passed to YOLO as `conf=`, then the
per-class threshold is applied as a post-filter.

### Soft-NMS Gaussian

Hard NMS would drop a legitimate second transformer if it overlapped IoU > 0.45
with the first one — common in dense settlements with cluster-mounted DTs.
Soft-NMS decays the score instead: `score *= exp(−IoU² · (1/σ))`.

**`soft_nms_sigma = 0.40`** — sharper Gaussian decay than the paper default
0.5, which we found helps closely-spaced small objects (wells, transformers)
keep their score against neighbours. Implemented in numpy on CPU because the
algorithm is sequential and per-step costs are tiny — a GPU implementation
adds one `.item()` sync per step (one per detection) for no gain.

### Context-gated tiling

Before running YOLO on a 1280 px tile, the tile bbox is intersected with the
pre-built `STRtree` of building/road polygons (from Stage 1, buffered by
`context_buffer_px = 128 px`). If no intersection, the tile is skipped
entirely. STRtree gives O(log N) per query vs O(N) for naive intersection.

In a rural village ortho this skips 40–70 % of tiles — the largest single
inference-time saving across the pipeline.

### YOLO training (Stage 2B)

`train_stage2b()` in `train/train_stage2.py` calls `YOLO.train(...)` with:

| Setting | Value | Reason |
|---|---|---|
| `imgsz` | 1280 | Matches inference tile size |
| `batch` | 4 | YOLOv11l at 1280 px fits batch=4 on the A4000 with `cache='ram'` (vs 2 with the older YOLOv9e) |
| `cache` | `ram` | Pre-decodes images once; trades RAM for I/O |
| `epochs` | 120 | Empirically converges between 80 and 110 |
| `cos_lr` | True | Cosine LR; smoother convergence |
| `mosaic` | 1.0 | Always-on mosaic |
| `close_mosaic` | 20 | Disables mosaic for the last 20 epochs so the model sees clean images before convergence (avoids mosaic-conditioned features) |
| `mixup` | 0.15 | Light mixup |
| `copy_paste` | 0.30 | Strong copy-paste — proven mAP lift for sparse small-object datasets |
| `flipud` | 0.5 | Aerial imagery is rotation-invariant; vertical flips are fine |
| `multi_scale` | True | YOLO resizes within ±50 % each batch |
| `dropout` | 0.1 | Light head dropout |
| `amp` | True | Auto-mixed precision |
| `workers` | `CFG.NUM_WORKERS` | Was 0 in an earlier version → serial dataloader → idle GPU |

If a checkpoint exists, the script does **not** use YOLO's built-in `resume`
because that reloads the *original* config (e.g. `coco.yaml`) instead of our
SVAMITVA YAML. It loads the weights manually and trains fresh.

---

## 9. Data Preprocessing

### Entry: `data/preprocessing.py → preprocess_folder()`

Converts raw SVAMITVA rasters + shapefiles into the training artefacts. Key
design choices:

**Strip-based memory-safe processing.** A typical SVAMITVA TIF is up to
213 734 × 112 836 px (~72 GB RGB). Loading it whole would OOM a 32 GB box.
Instead, per raster:

1. Read 4096-row horizontal strip RGB (~2.6 GB).
2. Burn the SHP mask for that strip (~0.9 GB).
3. Tile the strip → save 512 px patches.
4. Discard both arrays (GC).
5. Advance to the next strip.

**Peak RAM per raster: ~3.5 GB.**

**STRtree spatial index.** Each raster loads its SHPs once into a `shapely.STRtree`.
Per strip, only geometries that intersect the strip's bounding box are
rasterised — no brute-force iteration over thousands of polygons.

**Process-level parallelism.** Multiple rasters in parallel via
`ProcessPoolExecutor` (≤ 5 workers, ~17 GB RAM total). Within each raster,
tile writes use a `ThreadPoolExecutor` sized to `P-cores − 2`.

**Patch filtering.** Tiles where `foreground / total < min_fg_ratio (= 0.01)`
are dropped. Without this the training set is dominated by 99 % background
tiles that add noise without signal.

**Building crops (Stage 2A).** Uses `rasterio.windows.Window` to read only the
building's bounding box from disk — never loads the full strip just to extract
one crop. Uses `cv2.INTER_AREA` for downscaling (the only correct OpenCV
interpolation when shrinking).

**YOLO label generation (Stage 2B).** Object-centred tile strategy — each tile
is centred on an infrastructure cluster rather than grid-snapped. Class-specific
half-widths: `transformer=100`, `tank=80`, `well=40` px. **Negative tile
sampling** at 30 % of positive count teaches YOLO what non-infrastructure
looks like.

**`_to_uint8` normalisation.** 2nd–98th percentile per-channel stretch, with
**zero pixels excluded from percentile computation** so no-data regions don't
compress the valid radiometric range.

---

## 10. Training Pipeline

### Stage 1 — `train/train_stage1.py`

**Optimiser stack:**
- `AdamW(parameter_groups, lr=2e-4, weight_decay=1e-4)`
- Parameter groups: encoder (`lr × 0.1 = 2e-5`) vs decoder (`lr = 2e-4`),
  each split into `weight_decay` / `no_decay` for biases & norm weights.
- **SAM** wrapper available but **`use_sam=False`** by default for Stage 1.
  SAM is disabled because (a) it doubles per-step cost, (b) it forces
  `grad_accum=1`, which on the current 4 × 8 = 32 effective batch would
  drop us to a 4-sample effective batch unless `batch_size` is also raised.
  Enable only after Stage 1 training has stabilised and you have budget
  for a final SAM polish pass.

**Schedule:** `OneCycleLR(max_lr=per-group, pct_start=0.15, div_factor=25, final_div_factor=1e4)`.
`steps_per_epoch` is computed from `min(len(train_loader), MAX_STEPS_PER_EPOCH) / grad_accum`
so the schedule still terminates correctly when the cap is active.

**Effective batch:** `batch_size × grad_accum = 4 × 8 = 32`.
*Why 4×8 and not 8×4?* Same effective batch, but 4×8 halves the peak
activation memory — leaving headroom for SAM later and tolerating the
absence of `set_grad_checkpointing` on MiT encoders in `smp`.

**Multi-scale training:** 50 % of batches are randomly resized to one of
`ms_scales = (0.75, 1.0, 1.25)`. The resize is done with `F.interpolate`
*outside* the autocast context — the original wrapped it inside, forcing a
needless downcast/upcast through the bf16 pipeline.

**CutMix:** 20 % per-batch probability, applied at image+mask level. The mask
gets cut simultaneously. Skipped the unnecessary B×C×H×W clones — advanced
indexing already materialises a fresh tensor for the source slice.

**EMA:** decay 0.9998, on-GPU shadow buffers updated via
`torch._foreach_lerp_` (one fused kernel instead of N per parameter).
**Validation is wrapped in `try/finally`** so a `_validate` crash can't leave
the model swapped to EMA weights — the next training step would otherwise
`optimiser.step()` on EMA weights and feed those right back through `ema.update`.

**SWA:** `use_swa=True` from epoch `int(80 × 0.75) = 60`. At end of training,
SWA BN statistics are updated with one pass over the train loader. Final
SWA checkpoint saved as `stage1_swa.pth`. SWA typically gives +0.5–1.5 mIoU
over the single best checkpoint.

**Stratified train/val split.** Patches are sorted by foreground ratio into 4
quartile strata, then each stratum is split 85/15. This guarantees the
validation set reflects the full difficulty distribution rather than only easy
near-empty tiles. Mask reads use a `ThreadPoolExecutor` because they are
I/O-bound and OpenCV releases the GIL.

**Gradient clipping + spike guard:** `clip_grad_norm_(1.0)`; if the pre-clip
norm exceeds 10, the optimizer step is *skipped* and the scaler is updated
defensively. With bf16 the scaler is `None` so this is just `torch.*` ops.

**Validation TTA cadence:** `VAL_TTA_EVERY = 2` — full TTA every other epoch.
On non-TTA epochs validation uses a single forward (5–6× faster). The
best-checkpoint picker still sees high-fidelity signal half the time.

**GPU-side confusion matrix.** Validation accumulates a `(C, C)` int64 tensor
on the device via `torch.bincount(C·t + p, minlength=C²).reshape(C, C)`, then
does **one** CPU transfer at the end. The earlier code did a per-batch
`.cpu().numpy()` round-trip.

**Best/last checkpoint policy.**
- **Best**: only EMA weights + tiny metadata. Tiny file.
- **Last**: full model state + EMA + scheduler state every epoch, but the
  heavy optimizer state (~2× model size) only every 5 epochs. AdamW moments
  re-warm in a few steps, so this is a fair I/O tradeoff. Scheduler state is
  ~1.7 KB and is **always** saved — losing it would reset OneCycleLR back to
  warmup on resume.

**Early stop:** `patience = 18`. Decision broadcast from rank-0 in DDP so
worker ranks don't hang waiting for a step that won't come.

**DDP:** `utils/ddp.setup_ddp()` checks `WORLD_SIZE`; single-GPU is a no-op.
`make_loader` injects `DistributedSampler` automatically when DDP is on.
`set_epoch` is called every epoch for proper per-epoch shuffle.

**Post-validation `empty_cache()`:** TTA mega-batches (n_augs × B per scale)
leave the cached allocator full of large blocks that don't fit the next
training batch's shape. We release reserved-but-unused blocks before the
next epoch so peak fragmentation doesn't compound.

### Stage 2A — `train_stage2a()` in `train/train_stage2.py`

Same overall structure as Stage 1, with these specifics:

- **Optimiser:** AdamW + SAM (`use_sam=True`, `rho=0.05`, `adaptive=True`).
- **Schedule:** `SequentialLR(LinearLR warmup, CosineAnnealingWarmRestarts(T_0=⅓ of remaining))`.
- **Augmentation:** RandAugment, ColorJitter, RandomShadow, RandomFog,
  RandomSunFlare. Per-crop instance normalisation in `RooftopDataset` equalises
  brightness across villages with different sun angles.
- **MixUp + CutMix:** randomly chosen per batch (50/50). The combined-loss
  trick (one forward pass, two CE evaluations against `ya` and `yb`) keeps
  activation memory low.
- **VRAM auto-guard (NEW):** matches Stage 1's behaviour on ≤16.5 GB cards.
- **MAX_STEPS_PER_EPOCH cap (NEW):** parity with Stage 1.
- **Gradient-norm spike skip (NEW):** parity with Stage 1; the `did_step`
  flag prevents scheduler/EMA from advancing on a skipped step.

### Stage 2B — `train_stage2b()` in `train/train_stage2.py`

Thin wrapper around `YOLO.train(...)` (see Stage 2B section above).
`_write_yolo_yaml()` dynamically scans the dataset for present infrastructure
classes; it **keeps the configured class-name order** because YOLO label ids
were generated against `cfg["class_names"]` — sorting detected classes would
silently swap class ids.

### DDP

```bash
torchrun --nproc_per_node=2 train/launch_ddp.py
```

Only Stage 1 is DDP-aware in the current code. Stage 2A and 2B run on a
single GPU.

---

## 11. Inference Pipeline

### Entry: `inference/pipeline.py → GeoIntelPipeline`

**Constructor:**
- Loads all three models once.
- `channels_last` applied to both Stage 1 and Stage 2A (15–25 % NHWC gain on Ampere).
- `torch.compile` disabled by default (`COMPILE_ENABLED=False` because Triton is unavailable on Windows).
- Caches the cosine blending window once (originally recomputed every call).

**`run(tif_path, out_dir)`:**

```
1. Open TIF (or ECW → auto-convert to TIF) → read RGB → _to_uint8
2. _segment() → prob_map (C, H, W) float32
3. Optional DenseCRF refinement
4. argmax → seg_mask (H, W) uint8
5. Morphological cleanup
6. Write {prefix}_segmask.tif (fallback to timestamped path if QGIS holds the file)
7. mask_to_shapefile → per-class .shp + combined .gpkg
8. _classify_rooftops() using building.shp → merge → _building_rooftop.shp
9. _gather_context_polygons() → _detect() → _infrastructure.shp
```

**ECW handling:** `utils/ecw_compat.is_ecw / ecw_to_tif` runs at the start. It
tries (in order): the native rasterio ECW driver → OSGeo4W `gdal_translate` →
QGIS-bundled `gdal_translate` → `osgeo.gdal` Python bindings. Result is
written to a `tempfile.TemporaryDirectory` that's cleaned up at the end.

**`_segment()` tiling:**
- 512 px patches, 128 px overlap, batch=16.
- Edge patches reflect-padded via `cv2.copyMakeBorder` before transform (a small
  redundancy: the val transform also pads via `A.PadIfNeeded`, but with
  identical reflect semantics, so it's a no-op).
- Output blended via the cached cosine window; final divide by `count_map`.

**`_classify_rooftops()`:**
- Reuses the building GDF the caller already loaded (the original ignored the
  passed argument and re-read the SHP per call — non-trivial on 10K+
  buildings).
- Iterates with `gdf.geometry.values + index.to_numpy()` instead of
  `iterrows()` to avoid building a fresh Series per row.

**`_detect()` (Stage 2B):**
- Builds the STRtree once.
- Hands YOLO the BGR ndarray directly. The earlier path JPEG-encoded the
  patch, wrote it to a temp file, made YOLO re-read it, then deleted the
  file — per tile. Three disk I/Os + a lossy round-trip in the inference
  path, removed.
- Soft-NMS applied per class at the end.

**`clear_cuda_cache()` between stages** — Stage 2A's ConvNeXt-L blocks would
otherwise sit in the allocator while YOLO tries to grab its own 1280 px
tensors. Released so Stage 2B starts on a defragmented heap.

---

## 12. Post-Processing

### `utils/postprocess.py`

| Function | What it does | Notes |
|---|---|---|
| `apply_dense_crf` | Tiled CRF (1024 px tiles, 128 px overlap, cosine blend, up to `CRF_WORKERS` processes) | Falls back to identity if `pydensecrf` unavailable; falls back to serial if multiprocessing fails (Windows + pydensecrf can be flaky) |
| `clean_segmentation_mask` | Per-class morphology + watershed building separation | Detailed in Stage 1 section |
| `separate_touching_buildings` | Distance transform → `peak_local_max` → marker-controlled watershed | Falls back to identity if scipy/skimage missing |
| `mask_to_shapefile` | Rasterise each class layer → fix topology → simplify → explode → area-filter → write `.shp` per class + combined `.gpkg` | Skips background (class 0) |
| `clean_vector_geometries` | `make_valid` → simplify → explode multipart → drop sub-area polygons → drop invalid | Used by `mask_to_shapefile` internally |
| `merge_rooftop_labels` | Join Stage 2A predictions into the building GDF as `roof_pred` attribute | Output: `_building_rooftop.shp` |
| `detections_to_shapefile` | OBB → rotated rectangle polygons in geo-space; axis-aligned → centroid Points | Preserves YOLO OBB angle (clockwise pixel space → counter-clockwise geo space) |

---

## 13. Utilities Reference

### `utils/hardware.py`

| Symbol | Purpose |
|---|---|
| `setup(seed)` | Configures CUDA flags (TF32, cudnn.benchmark, Flash SDPA), thread counts for i9-13900K (8 P-cores), GDAL env vars, allocator config, RNG seeds. Idempotent. |
| `worker_init_fn(worker_id)` | DataLoader worker initialiser: caps PyTorch threads to 1, disables OpenCV's internal threading, pins each worker to a specific P-core (0–15). |
| `compile_model(model, mode, fullgraph)` | `torch.compile()` with graceful fallback. `fullgraph=True` is safe for ConvNeXt (no dynamic control flow); use `False` for transformers that have conditional ops. |
| `to_channels_last(model)` | Converts model to NHWC memory layout. |
| `cl_input(tensor)` | Converts a 4D input tensor to `channels_last` before forward. |
| `get_amp_context(dtype)` | Returns `(autocast_ctx, scaler)`. `scaler` is `None` for bf16 (no underflow) and a real `GradScaler` for fp16. |
| `maybe_backward(loss, scaler)` | Safe `loss.backward()` that handles both scaler-present and scaler-absent paths. |
| `maybe_step(opt, scaler, max_grad_norm, params)` | Optimizer step with optional grad-norm clipping; returns pre-clip norm for spike checks. |
| `EMA` | Exponential Moving Average. Shadow weights kept on GPU. `update()` uses `torch._foreach_lerp_` (one fused kernel). `apply_shadow()` / `restore()` swap for validation. |
| `vram_stats()` | One-liner GPU VRAM string (allocated / reserved / total). |
| `get_cuda_streams()` | Returns two `torch.cuda.Stream` objects for H2D/compute overlap (currently unused in the live training loop). |
| `clear_cuda_cache()` | `torch.cuda.empty_cache() + gc.collect()`. Called between stages. |

### `utils/sam.py` — Sharpness-Aware Minimisation

Wraps any base optimiser. Two phases per step:

1. **`first_step`** — compute gradient norm, scale a worst-case perturbation by
   `rho / (grad_norm + ε)`, add to weights. Persistent `old_p` buffers per
   parameter (allocated once, reused) and `torch._foreach_*` ops fold N
   per-parameter kernel launches into 1.
2. **`second_step`** — restore original weights, take the real optimizer step.

Incompatibilities (enforced at config-load time):
- **SAM + fp16 GradScaler** is forbidden — SAM's two-step protocol needs
  mid-step unscale + rescale which `GradScaler`'s public API doesn't
  expose. Use bf16 (default).
- **SAM forces `grad_accum=1`** — both forwards must be over the same batch.

### `utils/ddp.py`

| Symbol | Purpose |
|---|---|
| `setup_ddp(seed)` | Initialises `dist.init_process_group` if `WORLD_SIZE > 1`; backend = `nccl` on Linux/CUDA, else `gloo`. Returns a `DDPState`. |
| `wrap_ddp(model, state)` | `DistributedDataParallel` with `device_ids=[local_rank]` when CUDA. No-op when DDP off. |
| `make_sampler` | Returns `DistributedSampler` or `None`. |
| `make_loader` | DataLoader factory that wires the sampler in automatically. |
| `set_epoch(loader, epoch)` | Calls `sampler.set_epoch()` for proper per-epoch shuffle. |
| `is_main_process(state)` | True only on rank 0 (gates logging / checkpoint writes). |
| `cleanup_ddp()` | `dist.destroy_process_group()` if initialised. |

### `utils/checkpointing.py`

**`atomic_torch_save(payload, path)`** — writes to `path.tmp`, `fsync`s, then
`os.replace`s. Prevents corrupt half-written checkpoints on crash mid-save.

### `utils/metrics.py`

| Symbol | Purpose |
|---|---|
| `SegmentationMetrics` | Accumulates confusion matrix; computes per-class IoU/F1, mean IoU (foreground only, excludes class 0), pixel accuracy, foreground pixel accuracy |
| `compute_map` | VOC/COCO mAP: per class, per IoU threshold (default 0.5:0.05:0.95 + 0.5 separately) |
| `_box_iou`, `_voc_ap` | Helpers |

(`ClassificationMetrics` was removed — `_val_clf` uses `sklearn.metrics.classification_report` directly, the class had no callers.)

### `utils/logger.py`

| Symbol | Purpose |
|---|---|
| `configure_logging` | One-time configure: stdout handler, optional `LOG_FORMAT=json` for structured logs, optional file handler |
| `get_logger(name)` | Configured logger; idempotent |
| `crash_logged(log, action)` | Context manager: catches all exceptions, logs traceback + action context, re-raises. Optional `on_crash` recovery callback |
| `JsonFormatter` | Stable-key JSON output for downstream parsing |
| `format_exception` | One-line traceback formatter |

(`log_event` was removed — never called anywhere.)

### `utils/window.py`

**`cosine_window(size, overlap, power)`** — single source of truth for the
2-D cosine taper. Used by both Stage 1's `_segment()` and the DenseCRF tiler.
`power=2` gives a smooth C¹-continuous taper; no visible tile seams.

### `utils/ecw_compat.py`

ECW → GeoTIFF auto-conversion. Cascade of strategies:

1. Native rasterio ECW driver (rare — requires Hexagon-SDK-licensed GDAL build).
2. OSGeo4W `gdal_translate.exe` (recommended on Windows).
3. QGIS bundled `gdal_translate.exe`.
4. `osgeo.gdal` Python bindings.

Raises a clear `RuntimeError` with install instructions if all four fail.

---

## 14. Entry Points

| Script | Purpose | Usage |
|---|---|---|
| `run_pipeline.py` | Master CLI | `python run_pipeline.py --mode {preprocess,train_stage1,train_stage2,train_all,evaluate,infer,all} [--data_root DIR] [--tif FILE] [--out DIR]` |
| `inference/pipeline.py` | Single-image inference | `python inference/pipeline.py --tif file.tif --out ./out` |
| `infer_folder.py` | Batch inference on a folder | `python infer_folder.py --test_folder ./test --out_folder ./results` |
| `train/train_stage1.py` | Stage 1 training (standalone) | `python train/train_stage1.py` |
| `train/train_stage2.py` | Stage 2A / 2B training | `python train/train_stage2.py --stage {2a,2b,both}` |
| `train/launch_ddp.py` | Multi-GPU launcher | `torchrun --nproc_per_node=N train/launch_ddp.py` |
| `run_stage2b.py` | Stage 2B standalone runner | `python run_stage2b.py` |
| `export_models.py` | ONNX / TorchScript export | `python export_models.py` |
| `gui.py` | Desktop operator console | `python gui.py` (or `launch_gui.bat`) |
| `_setup_verify.py` | Post-install smoke test | `python _setup_verify.py` |
| `build.py` | PyInstaller binary build | `python build.py` |

---

## 15. GUI — Operator Console

**File:** `gui.py` (~2100 LOC)
**Framework:** PyQt6 + Matplotlib + OpenCV
**Theme:** custom dark/light palette with animated indicators

**Tabs:**

- **Pipeline Runner** — checkpoint pickers (S1/S2A/S2B), input TIF/ECW picker,
  output dir picker. Run button launches `inference/pipeline.py` as a
  `QProcess` subprocess (so the GUI itself doesn't need CUDA). Live log
  output with ANSI stripping; progress bar driven by log patterns.
- **Map Viewer** — opens the output segmentation mask TIF, renders with the
  4-class colour overlay (background black, building red, road grey, water
  blue). OpenCV decoded into a Matplotlib canvas; zoom/pan.
- **Results** — reads the `_all_features.gpkg` and displays per-class feature
  counts and class-distribution table.
- **Requirements** — checks Python deps and reports what's missing.

The GUI is decoupled from GPU resources: it calls `inference/pipeline.py` or
`infer_folder.py` via `QProcess` so it can run on any machine while training
runs elsewhere.

---

## 16. Configuration Reference

All values below are from the **current `config.py`**. Sections in the same
order as the source file.

### Paths

| Constant | Value | Purpose |
|---|---|---|
| `ROOT` | `<repo>` | Project root |
| `DATA_ROOT` | `ROOT/dataset` | Raw SVAMITVA root |
| `PATCH_DIR` | `DATA_ROOT/patches` | 512 px segmentation training tiles |
| `MASK_DIR` | `DATA_ROOT/patch_masks` | Corresponding class masks |
| `CROP_DIR` | `DATA_ROOT/building_crops` | Stage 2A rooftop crops, organised by class |
| `YOLO_DIR` | `DATA_ROOT/yolo_infra` | Stage 2B tiles + YOLO labels |
| `CKPT_DIR` | `ROOT/checkpoints` | Model checkpoints |
| `LOG_DIR` | `ROOT/logs` | Training logs |
| `OUT_DIR` | `ROOT/outputs/vectorized` | Inference outputs |
| `TRAIN_MASKS` | `DATA_ROOT/masks` | Full-raster masks (preprocessing intermediate) |

All directories are created on import via `d.mkdir(parents=True, exist_ok=True)`.

### Hardware / runtime

| Constant | Value | Why |
|---|---|---|
| `DEVICE` | `cuda` / `mps` / `cpu` | Auto-detect; CUDA covers both NVIDIA and AMD ROCm (PyTorch reports both as `cuda`) |
| `AMP_DTYPE` | `bfloat16` (CUDA) · `float16` (MPS) · `float32` (CPU) | bf16 is lossless for this workload on Ampere; MPS doesn't support bf16 in all ops |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:256` | Ampere-friendly allocator that grows segments on demand; eliminates long-run fragmentation from bf16 + ConvNeXt-L + TTA mega-batches. `max_split_size_mb` retained as defensive cap. |
| TF32 matmul + cudnn | `True` | Tensor-core fast path |
| cudnn.benchmark | `True` | Picks fastest kernel per input shape (we have fixed shapes) |
| `COMPILE_ENABLED` | `False` | `torch.compile` needs Triton, which isn't available on Windows |
| `COMPILE_MODE` | `"reduce-overhead"` | Lowest compile time; "max-autotune" gains ~5 % runtime at ~20 min compile cost |
| `FAST_TTA` | `False` | Full 24-pass TTA by default; flip for dev iteration |
| `NUM_WORKERS` | `8` | DataLoader workers (matches P-core count) |
| `PIN_MEMORY` | `True` | Enables non-blocking H2D DMA |
| `PREFETCH_FACTOR` | `4` | Batches prefetched per worker |
| `PERSISTENT_WORKERS` | `True` | Workers survive across epochs (saves spawn overhead) |
| `MAX_STEPS_PER_EPOCH` | `2000` | Caps epoch wall-clock on 30 GB+ datasets |
| `MAX_VAL_STEPS` | `300` | Caps val time per epoch; with `shuffle=False` it's a stable signal |
| `VAL_TTA_EVERY` | `2` | Full TTA every other epoch; faster single-forward val on the others |
| `CRF_WORKERS` | `4` | CRF tile parallelism (CPU heavy, RAM bounded) |

### Stage 1 — `STAGE1` dict

| Key | Value | Why |
|---|---|---|
| `num_classes` | `4` | Background + 3 foreground |
| `class_names` | `['background', 'building', 'road', 'waterbody']` | |
| `class_colors` | `[(0,0,0),(255,0,0),(128,128,128),(0,0,255)]` | Render palette |
| `arch` | `'MAnet'` | Sharper boundaries than Unet on this dataset (see Stage 1 section) |
| `encoder` | `'mit_b4'` | Strong texture features; 20 % faster than MiT-B5 |
| `encoder_weights` | `'imagenet'` | Standard pretraining |
| `in_channels` | `3` | RGB |
| `decoder_attention_type` | `'scse'` | Free attention boost on skip features |
| `patch_size` | `512` | Training and inference patch size |
| `patch_sizes` | `(512,)` | Multi-scale training pool (currently single scale; multi-scale handled via `ms_training` instead) |
| `overlap` | `128` | Inference tile overlap (25 % of patch) |
| `batch_size` | `4` | With 4×8 grad-accum = effective 32 (see Stage 1 section for why 4×8 over 8×4) |
| `grad_accum` | `8` | |
| `lr` | `2e-4` | Decoder learning rate |
| `encoder_lr_mult` | `0.1` | Encoder fine-tuned 10× more gently |
| `weight_decay` | `1e-4` | AdamW WD |
| `epochs` | `80` | Empirical convergence ≤ 60 with early stop |
| `warmup_epochs` | `3` | (Not directly used — OneCycleLR has its own warmup via `pct_start=0.15`) |
| `scheduler` | `'cosine'` | Annotation; actual scheduler is OneCycleLR |
| `use_sam` | `False` | See Stage 1 → "SAM disabled" note |
| `sam_rho` | `0.05` | Default Foret et al. value |
| `sam_adaptive` | `True` | Adaptive (per-parameter scale by weight magnitude) |
| `ms_training` | `True` | 50 %-prob random resize each batch |
| `ms_scales` | `(0.75, 1.0, 1.25)` | Tighter than the textbook `(0.5, 1.0, 1.5)` — keeps batch shapes within a single VRAM budget |
| `dice_weight` | `0.40` | Largest term — overlap is the most important target |
| `bce_weight` | `0.15` | CE (the key in config is `bce_weight` for historical reasons) |
| `focal_weight` | `0.15` | |
| `boundary_weight` | `0.15` | |
| `lovasz_weight` | `0.15` | |
| `touching_weight` | `0.10` | Smaller because it only affects building boundaries |
| `focal_gamma` | `2.0` | Standard |
| `class_weights` | `[0.20, 2.00, 5.00, 2.50]` | Road 5× because thin classes are under-represented in pixel count |
| `label_smoothing` | `0.08` | Slightly higher than typical 0.05 — combats SVAMITVA mask noise |
| `use_swa` | `True` | Free +0.5–1.5 mIoU |
| `swa_lr` | `2e-5` | Low, flat LR for SWA averaging |
| `swa_start_frac` | `0.75` | Start averaging from epoch 60 (75 % of 80) |
| `use_ema` | `True` | Decay 0.9998 |
| `ema_decay` | `0.9998` | |
| `cutmix_alpha` | `1.0` | Standard CutMix |
| `drop_path_rate` | `0.2` | MiT-B4 stochastic depth |
| `val_fraction` | `0.15` | 85/15 train/val split |
| `seed` | `42` | |
| `min_building_area_px` | `80` | Drop building blobs smaller than this in cleanup |
| `min_road_width_px` | `3` | Drives the road morphology kernel sizes |
| `polygon_min_area_px` | `{building:80, road:120, waterbody:160}` | Per-class vector-area filter |
| `polygon_simplify_tolerance` | `0.5` | Shapely simplify tolerance in pixel units |
| `crf_inference` | `True` | DenseCRF runs by default |
| `crf_iter` | `10` | Krähenbühl & Koltun convergence point |
| `neg_tile_ratio` | `0.15` | Negative tile sampling during preprocessing |
| `min_fg_ratio` | `0.01` | Drop patches with <1 % foreground from training set |

### Stage 2A — `STAGE2A` dict

| Key | Value | Why |
|---|---|---|
| `num_classes` | `4` | RCC / Tiled / Tin / Other |
| `class_names` | `['RCC', 'Tiled', 'Tin', 'Other']` | |
| `shp_roof_col` / `shp_roof_cols` | `'Roof_type'` / tuple of fallbacks | Tolerant to varied SVAMITVA column casing |
| `roof_type_map` | `ROOF_TYPE_MAP` | See Section 3 |
| `arch` | `'convnext_large'` | Best texture-classification backbone at 224 px |
| `pretrained` | `True` | ImageNet |
| `crop_size` | `224` | Model input |
| `min_crop_px` | `40` | Skip buildings smaller than 40 px (noise / artefacts) |
| `batch_size` | `32` | Halved to 16 by VRAM auto-guard if SAM is on |
| `lr` | `5e-5` | |
| `epochs` | `80` | |
| `label_smoothing` | `0.05` | |
| `mixup_alpha` | `0.4` | |
| `cutmix_alpha` | `1.0` | |
| `weight_decay` | `1e-4` | |
| `grad_accum` | `1` | |
| `tta_steps` | `8` | Per-scale fold count; total = 8 × 3 = 24 passes |
| `stage2a_conf_thresh` | `{RCC:0.45, Tiled:0.55, Tin:0.50, Other:0.40}` | Per-class calibration (see Stage 2A section) |
| `use_arcface` | `True` | Tighter inter-class margin |
| `arcface_s` | `30.0` | Standard scale |
| `arcface_m` | `0.55` | Tighter than 0.50; previously hardcoded ignoring this value, fixed |
| `use_sam` | `True` | Stage 2A is small enough to absorb SAM's 1.9× per-step cost |
| `sam_rho` | `0.05` | |
| `sam_adaptive` | `True` | |
| `use_randaugment` | `True` | |
| `randaugment_n` | `2` | |
| `randaugment_m` | `7` | Magnitude 7 — moderate |
| `drop_path_rate` | `0.4` | Heavy regularisation for ConvNeXt-L |
| `use_ema` | `True` | |
| `ema_decay` | `0.9995` | Slightly faster-adapting than Stage 1 (smaller model, fewer steps) |

### Stage 2B — `STAGE2B` dict

| Key | Value | Why |
|---|---|---|
| `class_names` | `['transformer', 'overhead_tank', 'well']` | Class id order is fixed; preprocessing writes labels against this order |
| `num_classes` | `3` | |
| `shp_infra_col` / `shp_infra_cols` | `'Utility_Ty'` / fallback tuple | |
| `infra_type_map` | `INFRA_TYPE_MAP` | See Section 3 |
| `model_variant` | `'yolo11l'` | ~2× faster than YOLOv9e at parity accuracy; C2PSA attention helps small objects |
| `use_obb` | `False` | Targets are rotationally symmetric / near-square → angle is undefined or ambiguous; Point centroids are what GIS users want |
| `obb_model_variant` | `'yolo11l-obb'` | Kept as a no-op fallback if OBB is ever re-enabled |
| `img_size` | `1280` | Matches inference tile size |
| `cache` | `'ram'` | Pre-decoded images cached in RAM |
| `batch_size` | `4` | YOLOv11l at 1280 px uses ~4.5 GB; doubled from 2 (yolov9e budget) for better gradient signal |
| `workers` | `NUM_WORKERS` | Parity with Stages 1/2A (was 0 → serial dataloader → idle GPU) |
| `dropout` | `0.1` | Light head dropout |
| `multi_scale` | `True` | YOLO resizes within ±50 % per batch |
| `epochs` | `120` | |
| `lr0` / `lrf` | `1e-3` / `0.01` | YOLO defaults |
| `warmup_epochs` | `3` | |
| `patience` | `20` | Early stop |
| `cos_lr` | `True` | |
| `mosaic` | `1.0` | Always-on |
| `close_mosaic` | `20` | Last 20 epochs mosaic-off for clean convergence |
| `hsv_h/s/v` | `0.015 / 0.5 / 0.3` | Standard YOLO HSV jitter |
| `degrees` | `15.0` | Rotation augmentation |
| `translate` | `0.1` | |
| `scale` | `0.6` | |
| `fliplr` / `flipud` | `0.5 / 0.5` | Aerial is rotation-invariant; vertical flip OK |
| `mixup` | `0.15` | Light mixup |
| `copy_paste` | `0.30` | Strong copy-paste — proven mAP lift for sparse small objects |
| `conf_thresh` | `0.10` | Default per-class fallback |
| `iou_thresh` | `0.60` | YOLO NMS IoU |
| `max_det` | `1000` | Per image |
| `overlap` | `512` | Inference tile overlap |
| `class_buffer_px` | `{transformer:100, overhead_tank:80, well:40}` | Per-class half-width in YOLO label generation |
| `context_classes` | `('building', 'road', 'waterbody')` | Used by context-gated tiling |
| `context_buffer_px` | `128` | Buffer around context polygons for tile inclusion |
| `neg_tile_ratio` | `0.3` | Negative tile sampling fraction |
| `soft_nms_sigma` | `0.40` | Sharper Gaussian decay than 0.5 paper default — helps closely-spaced small objects |
| `agnostic_nms` | `True` | Suppress cross-class duplicates |
| `use_sahi` | `True` | Sliced inference for small-object recall |
| `sahi_slice_size` | `512` | Smaller slices = better small-object recall |
| `sahi_overlap_ratio` | `0.45` | High overlap so edge objects are seen by ≥1 slice fully |
| `class_conf_thresh` | `{transformer:0.20, overhead_tank:0.12, well:0.10}` | See Stage 2B section |

---

## 17. A4000-Specific Optimization Notes

These are the choices motivated specifically by the RTX A4000 (Ampere, CC 8.6,
16 GB VRAM, PCIe 4.0 x16):

| Choice | Where | Why for A4000 specifically |
|---|---|---|
| `bfloat16` AMP | `config.AMP_DTYPE`, `utils/hardware.get_amp_context` | Ampere has native bf16 tensor cores; no `GradScaler` needed (8-bit exponent matches fp32) |
| Channels-last (NHWC) | `to_channels_last(module)` everywhere | Ampere tensor cores process NHWC convs 15–30 % faster than NCHW |
| TF32 matmul + cudnn | `setup()` | Ampere TF32 path: ~8× fp32 throughput, negligible accuracy loss |
| `cudnn.benchmark = True` | `setup()` | Fixed shapes → cuDNN picks the fastest kernel once |
| Flash Attention (SDPA) | `setup()` | Memory-efficient attention for transformer encoder (MiT-B4) |
| `expandable_segments:True` | `config.py`, `setup()` | PyTorch 2.1+ allocator that grows segments on demand; the single biggest fragmentation win on a 16 GB card running bf16 + ConvNeXt-L + TTA mega-batches |
| `max_split_size_mb:256` | Same | Defensive cap on legacy code paths |
| Effective batch 4×8 (Stage 1) | `STAGE1.batch_size / grad_accum` | Same effective batch as 8×4, half the activation peak — fits MAnet+MiT-B4 at 512 px without checkpointing and leaves SAM headroom |
| GPU-side confusion matrix | `train_stage1._validate` | Cuts B per-batch CPU round-trips down to one final transfer |
| Cosine window cached once | `inference/pipeline.py` | Constant given `(patch_size, overlap)` — was recomputed per call |
| Batched `_foreach_*` ops | `utils/sam.py`, `utils/hardware.EMA.update` | One kernel per op vs N per parameter — meaningful on 64 M-param MiT-B4 |
| `empty_cache()` after Stage 1 val | `train/train_stage1.py` | TTA mega-batches fragment the heap; release before next epoch |
| `clear_cuda_cache()` between Stage 2A & 2B | `train/train_stage2.py __main__`, `run_pipeline.py` | ConvNeXt-L blocks shouldn't sit in the allocator while YOLO grabs 1280 px tensors |
| VRAM auto-guard (Stage 1 + Stage 2A) | Both train loops | Halve batch on ≤16.5 GB when SAM is on; otherwise SAM's 2× peak would OOM |
| `MAX_STEPS_PER_EPOCH = 2000` | `config.py` | Caps wall-clock per epoch on 30 GB+ datasets |
| Gradient-norm spike skip | Both train loops | Skip step (and scheduler/EMA tick) when pre-clip norm > 10; prevents one bad batch from poisoning the run |
| EMA shadow on GPU + `_foreach_lerp_` | `utils/hardware.EMA` | No PCIe round-trips; two CUDA kernels instead of N |
| Soft-NMS on CPU/numpy | `models/stage2_models.soft_nms_gaussian` | Sequential algorithm — GPU version would `.item()`-sync per step (one per detection), CPU is faster |
| OpenMP thread cap in workers | `utils/hardware.worker_init_fn` | Without this, 8 DataLoader workers × `cpu_count()` OMP threads each thrash the i9 P-cores |
| CPU affinity to P-cores (0–15) | Same | Avoids work being preempted onto efficiency cores |
| `non_blocking=True` H2D transfers | Both train loops | Lets DMA overlap with the previous batch's backward when `pin_memory=True` |

---

## 18. Improvement History

Newest first.

### Latest — Stage 2B detector swap

**`config.STAGE2B`:**
- `model_variant`: `'yolov9e'` → **`'yolo11l'`** — ~2× faster at parity accuracy; C2PSA attention adds small-object recall; actively maintained
- `use_obb`: `True` → **`False`** — the three target classes (transformer / overhead_tank / well) are rotationally symmetric or near-square from above. OBB's angle regression is undefined for circles and 4-way ambiguous for squares; AABB outputs Point centroids which is what GIS users actually want
- `obb_model_variant`: `'yolov9e-obb'` → **`'yolo11l-obb'`** — kept as a no-op fallback for future re-enablement
- `batch_size`: `2` → **`4`** — YOLOv11l uses ~4.5 GB at imgsz=1280 (vs 7.2 GB for YOLOv9e); doubled batch fits and gives a cleaner gradient signal

**`models/stage2_models.py`:** updated stale `"yolov9e"` fallback strings in
`InfrastructureDetector._load()` to match the new default.

**`PROJECT_REFERENCE.md` checkpoint path:** updated from `stage2b_yolov9e/` to
`stage2b_yolo11l/` to match the auto-generated YOLO run directory.

### This session

**A4000 optimization pass**
- `config.py`: `PYTORCH_CUDA_ALLOC_CONF` now uses `expandable_segments:True,max_split_size_mb:256`. The expandable allocator is the documented Ampere-friendly mode in PyTorch 2.1+; it eliminates long-run fragmentation under bf16 + ConvNeXt-L + TTA.
- `utils/hardware.py`: same allocator hint mirrored inside `setup()` so unit tests / standalone scripts that don't import `config.py` first still get it before any CUDA allocation.
- `train/train_stage1.py`: `torch.cuda.empty_cache()` after each validation. TTA mega-batches leave the allocator full of large blocks the next epoch's training batches don't fit into.
- `train/train_stage2.py`: Stage 2A now has parity with Stage 1's safety rails:
  - VRAM auto-guard halves `batch_size` (local var, not cfg mutation) when SAM is on and VRAM ≤ 16.5 GB.
  - `MAX_STEPS_PER_EPOCH` cap honoured; `actual_steps` used as the loss/acc divisor so logs aren't underreported when the cap kicks in.
  - Gradient-norm spike skip; scheduler/EMA only advance on a real step via a `did_step` flag.
  - `clear_cuda_cache()` between Stage 2A and 2B in `__main__`.
  - `EMA` and `clear_cuda_cache` hoisted to top-level imports.
- `utils/sam.py`: `_grad_norm` skips the redundant `.to(device)` round-trip per parameter when all gradients already share one device.

**Bug fixes**
- `train/train_stage1.py`: validation was bracketed by `apply_shadow → _validate → restore` with no exception protection. If `_validate` raised (OOM during TTA, CUDA error, anything), the model stayed swapped to EMA shadow weights — the next training step would then `optimiser.step()` on EMA weights and feed those right back through `ema.update`, silently corrupting both the live model and the EMA. Now wrapped in `try/finally`.
- `inference/pipeline.py`: replaced `sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))` with a clean `from pathlib import Path` followed by `sys.path.insert(...)`.
- `run_pipeline.py`: `--mode train_stage2` ran Stage 2A back-to-back with 2B with no cache clear, leaving the heap full of ConvNeXt blocks when YOLO tried to grab 1280 px tensors. Added the same `clear_cuda_cache()` call that `--mode train_all` already had.

**Dead-code sweep**
- Unused top-level imports removed across: `utils/hardware.py` (`sys`), `utils/ecw_compat.py` (`Optional`), `models/stage2_models.py` (`List`), `train/train_stage1.py` (`Optional`, `DataLoader`, `cleanup_ddp`), `train/train_stage2.py` (`os`, `math`; folded redundant `pathlib` imports), `run_stage2b.py` (`os`), `tests/test_core_components.py` (`math`).
- Removed `_cosine_warmup` from `train/train_stage2.py` (defined but never called; Stage 2A uses `SequentialLR(LinearLR + CosineAnnealingWarmRestarts)` instead).
- Removed `log_event` from `utils/logger.py` (never called anywhere).
- Removed `ClassificationMetrics` from `utils/metrics.py` (`_val_clf` uses `sklearn.metrics.classification_report` directly).
- All 33 project files now parse-clean with zero unused top-level imports.

### Earlier session — accuracy + correctness pass

- `config.crf_iter`: 5 → 10 (DenseCRF needs ~10 iterations for boundary convergence).
- `config.min_fg_ratio`: 0.003 → 0.01 (filter near-empty training patches).
- `config.soft_nms_sigma`: 0.9 → 0.5 → 0.40 (0.9 barely suppressed; 0.40 is sharper than paper default and helps tightly-spaced small objects).
- `config.class_conf_thresh[well]`: 0.03 → 0.10 (eliminate false-positive well flood).
- Added per-class `stage2a_conf_thresh` calibration (was a single blanket 0.55).
- Added `FAST_TTA` module toggle.
- `inference._to_uint8`: min-max → 2nd–98th percentile (satellite outliers compressed valid range to 7/255; percentile gives 181/255).
- Removed `_segment()` `min(overlap, 128)` cap — full configured overlap is now used.
- `tta_predict` calls now controlled by `CFG.FAST_TTA` (was hardcoded `fast_tta=True`).
- `_classify_rooftops` confidence gate now reads per-class thresholds from config.
- `ArcFaceHead` now reads `m` and `s` from config (was hardcoded `m=0.50` — config value ignored for 11 commits).

### `improvements v0.1` batch (`67e91b4`)

- **Stage 1 model:** standardised on MAnet + MiT-B4; scSE on all decoder blocks; Lovász-Softmax added; instance-touching separation loss added; cosine-log Dice; 24-pass TTA with 0.875× added; deep-supervision-ready loss.
- **Stage 1 training:** multi-scale random resize 50 % per batch; stratified split by foreground ratio; OneCycleLR; VRAM auto-guard; gradient checkpointing on MiT-B4 (where supported); gradient spike guard.
- **Stage 1 inference:** spline window blending; shared `cosine_window` utility; tiled CRF with cosine blending; texture-aware bilateral; per-class CRF compatibility matrix; watershed building separation.
- **Stage 2A:** upgraded `convnext_base` → `convnext_large`; ArcFace head; deeper trunk; drop_path 0.4; 3-scale TTA; per-crop instance normalisation in `RooftopDataset`; MixUp + CutMix.
- **Stage 2B:** SAHI added; per-class confidence thresholds; configurable Soft-NMS sigma; OBB → rotated polygon export; class-specific bbox sizes in YOLO label generation.
- **Preprocessing:** strip-based processing; STRtree; ProcessPoolExecutor for rasters + ThreadPoolExecutor for tile writes; negative tile sampling; object-centred tile strategy; zero-pixel-excluded percentile stretch.
- **Infrastructure:** `utils/window.py`; `utils/logger.py` + `crash_logged`; atomic checkpoint save; multi-backend hardware (CUDA / ROCm / MPS / CPU); Flash Attention via SDPA.

### Earlier infrastructure batch (`fb618f4`)

- **SAM** (`utils/sam.py`) added.
- **DDP** (`utils/ddp.py`) added; single-GPU falls through transparently.
- **EMA** (`utils/hardware.EMA`) added.
- **SWA** added.
- Modular training pipelines as importable functions.
- Cross-platform installer scripts.
- Atomic checkpointing.

### `f0bd3da` — initial accuracy bump

- Road class weight raised.
- `arcface_m` set to 0.55 (initially ignored at the head; later fixed).
- SAHI overlap 0.30 → 0.40.
- Aggressive well threshold (later corrected).
- `agnostic_nms=True`.
- RandomFog + RandomSunFlare added to Stage 2A augmentation.

### `34880c2`, `91aad98` — PyQt6 GUI

Full operator console; subprocess-based pipeline launching; map viewer; results tab.

### `9f4ddb9` — SAM PyTorch 2.x fix

PyTorch 2.x renamed `_defaults` → `defaults`; fixed `utils/sam.py` to use the public attribute.

### Initial commit

Baseline pipeline (UNet, ConvNeXt-Base classifier, YOLO detector), simple
min-max normalisation, basic augmentation, AdamW only, single GPU.

---

*End of document. If you find a discrepancy between this file and the code,
the code wins — please update this document to match.*
