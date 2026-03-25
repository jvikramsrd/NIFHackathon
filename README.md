# Geo-Intel Hackathon — Two-Stage AI Pipeline
# AI-Based Feature Extraction from Drone Orthophotos

## Architecture Overview

```
INPUT: SVAMITVA drone orthophoto (GeoTIFF) + annotation shapefiles
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Semantic Segmentation                                │
│  Model   : UNet++ + EfficientNet-B4 backbone (ImageNet pretrained)│
│  Input   : 512×512 tiled patches (RGB)                          │
│  Output  : 4-class mask → Building / Road / Waterbody / BG     │
│  Loss    : Dice (0.6) + BCE (0.4), class-weighted               │
│  Post    : Dense CRF refinement + morphological cleanup         │
│  Vectorise: raster mask → SHP / GeoPackage polygons            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
┌──────────────────────┐              ┌─────────────────────────┐
│  STAGE 2A            │              │  STAGE 2B               │
│  Rooftop Classifier  │              │  Infrastructure Detector │
│  EfficientNet-B2     │              │  YOLOv8-m               │
│  Classes:            │              │  Classes:               │
│    RCC / Tiled /     │              │    Distribution Xfmr    │
│    Tin / Other       │              │    Overhead Tank        │
│  TTA + MixUp         │              │    Well                 │
└──────────┬───────────┘              └──────────┬──────────────┘
           │                                     │
           ▼                                     ▼
    building_rooftop.shp              infrastructure_points.shp
```

## Deliverables (per hackathon requirements)
- ✅ Building footprints (segmentation mask → polygon SHP)
- ✅ Rooftop material classification (RCC / Tiled / Tin / Other)
- ✅ Road network extraction (SHP)
- ✅ Waterbody extraction (SHP)
- ✅ Infrastructure point detection (transformer / overhead tank / well)
- ✅ ≥95% accuracy target (tracked via mIoU, pixel acc, class F1)
- ✅ Optimised: AMP training, EfficientNet backbone, TTA inference

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# Optional CRF:
pip install pydensecrf2
```

### 2. Organise your data
```
dataset/
├── CG_450163/
│   ├── village.tif          # orthophoto
│   ├── buildings.shp        # building footprints with roof_type attribute
│   ├── roads.shp
│   ├── waterbodies.shp
│   └── infrastructure.shp   # with infra_type attribute
├── CG_451189/
│   └── ...
└── ...
```

Edit `config.py` to match your SHP attribute column names:
- `STAGE1["shp_class_col"]`  — column that holds class label in SHP
- `STAGE2A["shp_roof_col"]`  — column for rooftop material
- `STAGE2B["shp_infra_col"]` — column for infrastructure type

### 3. Run full pipeline
```bash
# All steps in one go:
python run_pipeline.py --mode all --data_root ./dataset

# Or step by step:
python run_pipeline.py --mode preprocess --data_root ./dataset
python run_pipeline.py --mode train_stage1
python run_pipeline.py --mode train_stage2
python run_pipeline.py --mode evaluate

# Inference on a new village:
python run_pipeline.py --mode infer --tif /path/to/new_village.tif --out ./outputs/new_village
```

## Config Key Parameters (config.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 512 | Tile size for segmentation |
| `arch` | UnetPlusPlus | smp segmentation architecture |
| `encoder` | efficientnet-b4 | Backbone |
| `epochs` (stage1) | 80 | Training epochs |
| `lr` | 1e-4 | AdamW learning rate |
| `crf_inference` | True | Dense CRF post-processing |

## File Structure
```
geo_intel/
├── config.py                  ← all hyperparameters here
├── run_pipeline.py            ← master script
├── requirements.txt
├── data/
│   ├── preprocessing.py       ← TIF→patches, SHP→masks, YOLO labels
│   └── dataset.py             ← PyTorch datasets + augmentations
├── models/
│   ├── stage1_segmentation.py ← UNet++ + DiceBCE + TTA
│   └── stage2_models.py       ← EfficientNet-B2 + YOLOv8
├── train/
│   ├── train_stage1.py        ← Stage 1 training loop (AMP, warmup, early stop)
│   └── train_stage2.py        ← Stage 2A + 2B training
├── inference/
│   └── pipeline.py            ← end-to-end inference on new TIF
└── utils/
    ├── metrics.py             ← mIoU, pixel acc, mAP@0.5
    └── postprocess.py         ← CRF, morphology, vectorisation
```

## Accuracy Strategy 
1. **Pretrained backbone** — EfficientNet-B4 (ImageNet weights → strong feature extraction)
2. **UNet++ skip connections** — denser feature reuse vs vanilla UNet
3. **8-fold TTA** — averages predictions over rotations + flips
4. **Dense CRF** — sharpens boundary predictions
5. **Weighted loss** — compensates for class imbalance (rare waterbodies / infra)
6. **MixUp + label smoothing** — regularises rooftop classifier
7. **YOLOv8-m** — state-of-the-art small-object detection for infrastructure
