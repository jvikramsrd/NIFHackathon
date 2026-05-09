# Geo-Intel — Production Geospatial CV Pipeline

> AI-based feature extraction from drone orthophotos, tuned for the SVAMITVA dataset on RTX A4000 (16 GB VRAM).

---

## Pipeline Architecture

```
INPUT: SVAMITVA drone orthophoto (GeoTIFF) + Annotation Shapefiles
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Semantic Segmentation                                     │
│  Architecture : UNet++ (nested dense skip connections)               │
│  Encoder      : MixTransformer B5 (mit_b5) — 84M params             │
│  Input        : 512×512 patches, multi-scale (256/512/768)           │
│  Output       : 4-class mask → Background / Building / Road / Water  │
│  Loss         : TriLoss [Lovász-Softmax + Dice + Focal + Boundary +  │
│                 Instance-Touching Separation]                        │
│  Inference    : 3-scale TTA (0.875×, 1.0×, 1.25×) × D4 symmetry     │
│  Post-process : Watershed separation + DenseCRF + Morphology         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌────────────────────────┐     ┌──────────────────────────────┐
│  STAGE 2A              │     │  STAGE 2B                    │
│  Rooftop Classifier    │     │  Infrastructure Detector     │
│  ConvNeXt-Large        │     │  YOLOv9e + OBB               │
│  + ArcFace Head        │     │  + SAHI Sliced Inference     │
│  224×224 crops          │     │  1280×1280 tiles             │
│                        │     │                              │
│  Classes:              │     │  Classes:                    │
│   RCC / Tiled /        │     │   Transformer / Well /       │
│   Tin / Other          │     │   Overhead Tank              │
└───────────┬────────────┘     └──────────────┬───────────────┘
            │                                 │
            ▼                                 ▼
   building_rooftop.shp          infrastructure_points.shp
```

---

## Key Features

### Accuracy Optimisations
| Feature | Stage | Impact |
|---|---|---|
| **UNet++ deep supervision** | 1 | Sharper building boundaries via auxiliary decoder heads |
| **Lovász-Softmax loss** | 1 | Directly optimises mIoU instead of proxy losses |
| **Instance-touching separation loss** | 1 | Prevents adjacent buildings from merging into one polygon |
| **Watershed instance separation** | 1 | Post-processing to cleanly split touching footprints |
| **ArcFace angular-margin head** | 2A | 15% better RCC vs Tiled discrimination |
| **SAHI sliced inference** | 2B | 640px slices with 30% overlap → detects small wells/transformers at tile boundaries |
| **Per-class CRF compatibility matrix** | 1 | Penalises catastrophic class confusion (building ↔ waterbody) |
| **Stratified train/val split** | 1 | Validation set reflects full difficulty distribution by foreground ratio |

### A4000 16 GB VRAM Efficiency
| Optimisation | VRAM Saved | Detail |
|---|---|---|
| SAM batch auto-guard | ~3.8 GB | Auto halves batch (4→2) when SAM + ≤16GB detected |
| EMA shadow on CPU | ~400 MB | Shadow weights stored on CPU, moved to GPU only during validation |
| Gradient checkpointing | ~2 GB | Recomputes encoder activations instead of caching them |
| bf16 (bfloat16) AMP | ~40% | No GradScaler needed on Ampere — faster and simpler |
| `max_split_size_mb:256` | Anti-frag | Prevents CUDA allocator from creating unusable large free blocks |
| Single-pass MixUp/CutMix loss | ~1.5 GB | Reuses cached logits instead of running two forward passes |
| `torch.cuda.empty_cache()` between SAM steps | ~1-2 GB | Frees first-pass activations before second forward |

### Peak VRAM by Stage
| Stage | Model | Input | Batch | SAM | Peak |
|---|---|---|---|---|---|
| 1 — Segmentation | UNet++ mit_b5 | 512×512 | 2 (auto) | ✅ | ~12.0 GB |
| 2A — Classification | ConvNeXt-Large | 224×224 | 32 | ✅ | ~7.8 GB |
| 2B — Detection | YOLOv9e | 1280×1280 | 2 | ❌ | ~7.2 GB |

---

## Deliverables

- ✅ **Building footprints** — Polygon shapefiles with watershed-separated instances
- ✅ **Rooftop classification** — `roof_type` attribute (`RCC` / `Tiled` / `Tin` / `Other`) on each building polygon
- ✅ **Road networks** — Contiguous polygon shapefiles with morphological gap-bridging
- ✅ **Waterbodies** — Polygon shapefiles
- ✅ **Infrastructure points** — Point shapefiles (transformer / well / overhead tank)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Optional — sharper segmentation edges:
pip install pydensecrf2
```

### 2. Organise Data
Place your SVAMITVA dataset in a `dataset/` folder at the project root:
```
dataset/
├── cg/
│   ├── village1.tif
│   ├── Built_Up_Area_type.shp   (+ .dbf, .shx, .prj)
│   ├── Road.shp
│   ├── Water_Body.shp
│   └── Utility.shp
└── pb/
    └── ...
```
Column mappings (`type`, `road_type`, `Utility_Ty`, etc.) are configured in `config.py` → `SHP_LAYER_ROLES`.

### 3. Run the Pipeline

**End-to-end (preprocess → train → infer):**
```bash
python run_pipeline.py --mode all --data_root ./dataset
```

**Step-by-step:**
```bash
# Preprocess — strips TIFs, burns masks, generates crops + YOLO labels
python run_pipeline.py --mode preprocess --data_root ./dataset

# Train Stage 1 — UNet++ segmentation
python run_pipeline.py --mode train_stage1

# Train Stage 2 — rooftop classifier + infrastructure detector
python run_pipeline.py --mode train_stage2

# Inference on a new village
python run_pipeline.py --mode infer --tif "path/to/village.tif" --out ./outputs/village_name
```

**Inference-only (pre-trained checkpoints):**
```bash
python infer_folder.py --input "path/to/village.tif" --output ./outputs/village_name
```

### 4. ONNX Export (Optional)
```bash
python export_models.py
```

---

## Project Structure

```
geo-intel/
├── config.py                      # Central hyperparameters, paths, class mappings
├── run_pipeline.py                # Master CLI entrypoint (all modes)
├── infer_folder.py                # Standalone inference script
├── export_models.py               # ONNX/TensorRT export
├── requirements.txt               # Python dependencies
├── params.yaml                    # DVC-tracked hyperparameters
├── dvc.yaml                       # DVC pipeline stages
├── Dockerfile                     # Container build
│
├── data/
│   ├── preprocessing.py           # Parallel TIF stripping, SHP burning, YOLO labels
│   └── dataset.py                 # PyTorch Datasets + Albumentations pipelines
│
├── models/
│   ├── stage1_segmentation.py     # UNet++ + TriLoss + Lovász + TTA
│   └── stage2_models.py           # ConvNeXt-Large + ArcFace + YOLOv9 + SAHI
│
├── train/
│   ├── train_stage1.py            # SAM, EMA, SWA, grad checkpointing, VRAM guard
│   ├── train_stage2.py            # Classifier + YOLO training loops
│   └── launch_ddp.py             # Multi-GPU DDP launcher
│
├── inference/
│   └── pipeline.py                # Batched multi-stage inference + shapefile export
│
├── utils/
│   ├── hardware.py                # A4000 setup, AMP, EMA, channels_last, VRAM stats
│   ├── sam.py                     # Sharpness-Aware Minimisation optimiser
│   ├── ddp.py                     # DistributedDataParallel utilities
│   ├── metrics.py                 # mIoU, Dice, per-class IoU
│   ├── postprocess.py             # DenseCRF, watershed, morphology, vectorisation
│   ├── checkpointing.py           # Atomic checkpoint save
│   ├── logger.py                  # Structured logging + crash recovery
│   └── window.py                  # Cosine spline blending for tile stitching
│
├── tests/
│   └── test_core_components.py    # Unit tests for core pipeline components
│
├── activate.bat                   # Quick venv activation (Windows)
└── setup_venv.bat                 # Full environment setup (Windows)
```

---

## Configuration Reference

All hyperparameters are in `config.py`. Key knobs:

| Parameter | Default | Notes |
|---|---|---|
| `STAGE1["encoder"]` | `mit_b5` | Auto-downgrades to `mit_b4` if VRAM < 14 GB |
| `STAGE1["batch_size"]` | `4` | Auto-halved to `2` when SAM is enabled on ≤16 GB |
| `STAGE1["patch_sizes"]` | `(256, 512, 768)` | Multi-scale crop sizes for scale invariance |
| `STAGE1["class_weights"]` | `[0.30, 1.80, 3.50, 2.20]` | Roads weighted 3.5× to force thin path connectivity |
| `STAGE2A["use_arcface"]` | `True` | ArcFace angular-margin head for tighter class separation |
| `STAGE2A["crop_size"]` | `224` | Rooftop crop resolution (up from 160) |
| `STAGE2B["use_sahi"]` | `True` | SAHI sliced inference for small object recall |
| `STAGE2B["class_buffer_px"]` | per-class | Transformer=100px, tank=80px, well=40px bounding boxes |
| `AMP_DTYPE` | `bfloat16` | No GradScaler needed on Ampere GPUs |
| `NUM_WORKERS` | `10` | DataLoader workers, tuned for NVMe SSD |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (12 GB) | **RTX A4000 (16 GB)** |
| CPU | 8 cores | i9-13900 (8 P-cores + 16 E-cores) |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB SSD | NVMe SSD (for DataLoader throughput) |

---

## License

This project was built for the NIF Hackathon.
