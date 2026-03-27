# Geo-Intel Hackathon — Two-Stage AI Pipeline
# AI-Based Feature Extraction from Drone Orthophotos

This repository contains a highly optimized, high-accuracy geospatial deep learning pipeline designed to automatically extract buildings, road networks, waterbodies, rooftop materials, and utility infrastructure from massive high-resolution drone orthophotos (like the SVAMITVA dataset).

It has been heavily engineered for maximum throughput on an **NVIDIA RTX A4000 (16 GB VRAM)** and modern multi-core CPUs.

## Architecture Overview

```text
INPUT: SVAMITVA drone orthophoto (GeoTIFF) + Annotation Shapefiles
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Semantic Segmentation                                  │
│  Model   : UNet + MixTransformer B4 (mit_b4)                      │
│  Input   : 512×512 tiled patches (RGB)                            │
│  Output  : 4-class mask → Background / Building / Road / Water    │
│  Loss    : Tri-Loss [Dice + BCE + Focal] with heavy road weights  │
│  Infer   : Batched Fast Multi-Scale TTA (Rotations + Zoom)        │
│  Post    : Dense CRF + Elliptical Morphology + Spline Blending    │
│  Export  : Vectorisation to .shp / .gpkg                          │
└──────────────────────────────┬────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
┌─────────────────────────┐           ┌─────────────────────────────┐
│  STAGE 2A               │           │  STAGE 2B                   │
│  Rooftop Classifier     │           │  Infrastructure Detector    │
│  Model: ConvNeXt-Large  │           │  Model: YOLOv8-x            │
│  Classes:               │           │  Classes:                   │
│    RCC / Tiled /        │           │    Transformer / Well /     │
│    Tin / Other          │           │    Overhead Tank            │
│  Method:                │           │  Method:                    │
│    Multi-Scale TTA +    │           │    1280px Overlapping Tiles │
│    Confidence Threshold │           │    + Gaussian Soft-NMS      │
└──────────┬──────────────┘           └──────────────┬──────────────┘
           │                                         │
           ▼                                         ▼
   building_rooftop.shp                  infrastructure_points.shp
```

## Key Engineering & Optimizations (A4000 Edition)
- ⚡ **Parallel CPU Preprocessing**: Utilizes `ProcessPoolExecutor` to crunch through multiple 70GB+ GeoTIFFs simultaneously, dropping dataset preparation time by over 80%.
- ⚡ **Maximized Tensor Cores**: Inference pipelines group image chips into massive batches (`batch_size=16` for segmentation, `64` for roofs) keeping the RTX A4000 at 100% utilization.
- 🎯 **Fast Multi-Scale TTA**: Segmentation and rooftop classification use a highly efficient Test-Time Augmentation scheme that evaluates multiple scales (1.0x and 1.25x zoom) and rotations to drastically sharpen boundaries and recognize tiny textures, without the redundant overhead of flip-passes.
- 🎯 **Seamless Tile Stitching**: Employs a 2D Cosine Spline Window to seamlessly blend 512x512 inference patches together, completely eliminating grid-like tile artifacts.
- 🎯 **Advanced Morphology**: Uses `cv2.MORPH_ELLIPSE` kernels instead of squares to gracefully bridge tree-canopy gaps in winding roads without introducing jagged, blocky artifacts.
- 🎯 **Soft-NMS for Infrastructure**: Replaces standard YOLO NMS with Gaussian Soft-NMS, preventing closely-packed utility objects (like two transformers on one pole) from being accidentally deleted.

## Deliverables
- ✅ **Building footprints**: Polygon shapefiles.
- ✅ **Rooftop classification**: Appended as a direct attribute column (`RCC`, `Tin`, `Tiled`, `Other`) into the building polygons. Unsure roofs default to `Other`.
- ✅ **Road networks**: Smooth, contiguous polygon shapefiles spanning the village.
- ✅ **Waterbodies**: Polygon shapefiles.
- ✅ **Infrastructure points**: Point geometry shapefiles marking bounding-box centroids.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# Optional but highly recommended for crisp segmentation edges:
pip install pydensecrf2
```

### 2. Organize your SVAMITVA data
Ensure your dataset is structured inside a `dataset/` folder at the root of the project, containing the `cg/` and `pb/` state folders.
```text
dataset/
├── cg/
│   ├── village1.tif
│   ├── Built_Up_Area_type.shp  (Buildings + Roof_type)
│   ├── Road.shp                (Roads)
│   ├── Water_Body.shp          (Water)
│   └── Utility.shp             (Infrastructure points)
└── pb/
    └── ...
```
*(The pipeline automatically maps standard SVAMITVA column headers like "type", "road_type", and "Utility_Ty" based on the mappings in `config.py`)*

### 3. Run the Pipeline

**End-to-End Execution (All steps):**
```bash
python run_pipeline.py --mode all --data_root ./dataset
```

**Step-by-Step Execution:**
```bash
# 1. Preprocess (Strips TIFs, burns masks, parallelized across CPU cores)
python run_pipeline.py --mode preprocess --data_root ./dataset

# 2. Train Models (Uses EMA, Stochastic Weight Averaging, and OneCycleLR)
python run_pipeline.py --mode train_stage1
python run_pipeline.py --mode train_stage2

# 3. Evaluate Metrics (mIoU, F1 Score)
python run_pipeline.py --mode evaluate
```

**Run Lightning-Fast Inference on a New Village:**
```bash
python run_pipeline.py --mode infer --tif "C:/path/to/new_village.tif" --out ./outputs/village_name
```
*Outputs will be saved in `./outputs/village_name/` as ready-to-use `.shp` and `.gpkg` files!*

## Configuration (`config.py`)
All critical hyper-parameters, hardware allocations, and class mappings are centrally managed in `config.py`. 
- **`STAGE1['class_weights']`**: Dictates loss priority. Roads are currently weighted heavily (`3.0`) to force the model to connect thin, tree-covered paths.
- **`STAGE2B['overlap']`**: Defines the sliding window overlap for YOLO. Set to `512` for large 1280px tiles to ensure massive overhead tanks are never sliced in half.
- **`NUM_WORKERS`**: Controls PyTorch DataLoader threading. Set to 10 by default for fast NVMe SSDs.

## File Structure
```text
geo_intel_a4000/
├── config.py                  ← Central hyperparameters & path management
├── run_pipeline.py            ← Master CLI entrypoint
├── requirements.txt           
├── data/
│   ├── preprocessing.py       ← Parallel TIF stripping, Shapefile burning, YOLO crops
│   └── dataset.py             ← PyTorch Dataset classes + Albumentations pipelines
├── models/
│   ├── stage1_segmentation.py ← UNet + mit_b4 + TriLoss + Fast TTA
│   └── stage2_models.py       ← ConvNeXt-Large (Mixup) + YOLOv8x (Soft-NMS)
├── train/
│   ├── train_stage1.py        ← SWA, EMA, Gradient Accumulation, AMP
│   └── train_stage2.py        ← Classifier & YOLO training loop
├── inference/
│   └── pipeline.py            ← Batched multi-stage inference & shapefile generation
└── utils/
    ├── metrics.py             ← mIoU, Dice, Pixel Accuracy calculations
    └── postprocess.py         ← Dense CRF, Elliptical Morphology, Vectorisation
```

## Accuracy Strategy 
1. **Pretrained backbone** — MixTransformer B4 (mit_b4) (ImageNet weights → strong feature extraction)
2. **UNet skip connections** — denser feature reuse
3. **Fast Multi-Scale TTA** — averages predictions over rotations + zoom
4. **Dense CRF** — sharpens boundary predictions
5. **Weighted loss** — compensates for class imbalance (rare waterbodies / infra)
6. **MixUp + label smoothing** — regularises rooftop classifier
7. **YOLOv8-x** — state-of-the-art small-object detection for infrastructure