# Geo-Intel — Production Geospatial CV Pipeline

> AI-based feature extraction from drone orthophotos, tuned for the SVAMITVA dataset.  
> Supports NVIDIA CUDA · AMD ROCm · Apple Silicon MPS · CPU.

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
│  224×224 crops         │     │  1280×1280 tiles             │
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

## Installation

### Option A — Pre-built binary (recommended for new users)

Download the binary for your platform from the [GitHub Releases](../../releases) page:

| Platform | File |
|----------|------|
| Windows 10/11 (x64) | `GeoIntel-windows-x64.zip` |
| macOS 13+ Intel | `GeoIntel-macos-intel.zip` |
| macOS 14+ Apple Silicon (M1/M2/M3/M4) | `GeoIntel-macos-arm64.zip` |
| Linux x64 (Ubuntu 22.04+) | `GeoIntel-linux-x64.zip` |

**The binary is only the GUI shell.** You still need a Python environment with PyTorch + pipeline deps installed (see Option B below) for training and inference. The GUI auto-detects your active Python.

```
# After unzipping:

Windows:    GeoIntel\GeoIntel.exe
macOS:      open GeoIntel.app            (or GeoIntel/GeoIntel from terminal)
Linux:      ./GeoIntel/GeoIntel
```

---

### Option B — Install from source (for training / development)

#### Prerequisites

- Python 3.10 or 3.11
- Git

#### 1. Clone

```bash
git clone https://github.com/your-org/geo-intel.git
cd geo-intel
```

#### 2. Install PyTorch for your hardware

Pick **one** of the following:

**NVIDIA GPU (CUDA 12.1) — RTX 20xx / 30xx / 40xx, A-series, Quadro:**
```bash
pip install -r requirements-torch-cuda.txt
```

**NVIDIA GPU (CUDA 11.8) — GTX 10xx / 20xx, older Ampere:**
```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU (ROCm 6.2) — RX 6600 XT or newer, Linux only:**
```bash
pip install -r requirements-torch-rocm.txt
```

**Apple Silicon (M1 / M2 / M3 / M4) — Metal MPS acceleration:**
```bash
pip install -r requirements-torch-cpu.txt    # standard wheel includes MPS
```

**CPU-only / unknown GPU:**
```bash
pip install -r requirements-torch-cpu.txt
```

#### 3. Install everything else

```bash
pip install -r requirements.txt
```

**Geospatial stack** (rasterio / fiona / geopandas):
```bash
# Recommended — handles ECW driver + correct GDAL binaries:
conda install -c conda-forge libgdal-ecw rasterio fiona geopandas

# Pip-only alternative (no ECW):
pip install rasterio fiona geopandas
```

**Optional — dense CRF boundary refinement** (requires C++ build tools):
```bash
pip install pydensecrf2
```

#### 4. One-shot installer scripts

**macOS / Linux** — auto-detects NVIDIA / AMD / Apple Silicon:
```bash
chmod +x install.sh
./install.sh
```

**Windows** — auto-detects NVIDIA, prompts for manual ROCm install:
```bash
setup_venv.bat
```

---

### Option C — pip install

```bash
# Install PyTorch first (see step 2 above), then:
pip install .
```

This installs two console commands:
- `geo-intel-gui` — launch the desktop GUI
- `geo-intel-pipeline --mode ...` — CLI pipeline runner

---

## Hardware Support

| Accelerator | Platform | PyTorch Backend | AMP dtype |
|-------------|----------|-----------------|-----------|
| NVIDIA RTX 30xx / 40xx (Ampere) | Win / Linux | CUDA 12.1 | bfloat16 |
| NVIDIA RTX 20xx / older | Win / Linux | CUDA 11.8 | bfloat16 |
| AMD RX 6600 XT+ (RDNA 2/3) | Linux | ROCm 6.2 | bfloat16 |
| Apple M1 / M2 / M3 / M4 | macOS arm64 | MPS | float16 |
| Intel / AMD CPU | All | CPU | float32 |

> **ROCm on Windows:** ROCm does not support Windows. AMD GPU users on Windows must use CPU mode or WSL2.  
> **MPS note:** `torch.compile` and some fused kernels are disabled on MPS automatically.

### Peak VRAM by Stage

| Stage | Model | Input | Batch | NVIDIA A4000 | Apple M2 Max (96 GB unified) |
|-------|-------|-------|-------|--------------|-------------------------------|
| 1 — Segmentation | UNet++ mit_b5 | 512×512 | 2 | ~12.0 GB | ~18 GB unified |
| 2A — Classification | ConvNeXt-Large | 224×224 | 32 | ~7.8 GB | ~8 GB unified |
| 2B — Detection | YOLOv9e | 1280×1280 | 2 | ~7.2 GB | ~9 GB unified |

---

## Quick Start

### Desktop GUI
```bash
python gui.py
```

Tabs:
- **Pipeline Runner** — preprocess, train, evaluate, infer with live log + progress
- **Map Viewer** — side-by-side TIF vs. segmentation overlay with opacity slider
- **Results** — metrics table + bar chart, auto-reads `outputs/results.json`

### CLI

```bash
# Preprocess — strips TIFs, burns masks, generates crops + YOLO labels
python run_pipeline.py --mode preprocess --data_root ./dataset

# Train all stages
python run_pipeline.py --mode train_all

# Evaluate (writes outputs/results.json)
python run_pipeline.py --mode evaluate

# Inference on a new village
python run_pipeline.py --mode infer \
  --tif "path/to/village.tif" \
  --out ./outputs/village_name

# End-to-end
python run_pipeline.py --mode all --data_root ./dataset
```

### Build a binary locally

```bash
pip install pyinstaller>=6.5.0
python build.py

# Output:
#   dist/GeoIntel/GeoIntel.exe     (Windows)
#   dist/GeoIntel.app              (macOS)
#   dist/GeoIntel/GeoIntel         (Linux)
```

---

## Data Structure

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

Column mappings (`type`, `road_type`, `Utility_Ty`, etc.) are in `config.py` → `SHP_LAYER_ROLES`.

---

## Project Structure

```
geo-intel/
├── config.py                      # Hardware, paths, hyperparameters, class maps
├── run_pipeline.py                # Master CLI (preprocess/train/evaluate/infer)
├── gui.py                         # Desktop operator console (PyQt6)
├── infer_folder.py                # Standalone inference script
├── export_models.py               # ONNX export
├── build.py                       # Binary build script (PyInstaller)
├── geo_intel.spec                 # PyInstaller spec
├── pyproject.toml                 # pip-installable package definition
├── requirements.txt               # Base deps (no torch — platform-specific)
├── requirements-torch-cuda.txt    # NVIDIA CUDA torch
├── requirements-torch-rocm.txt    # AMD ROCm torch
├── requirements-torch-cpu.txt     # CPU / Apple MPS torch
├── install.sh                     # macOS + Linux one-shot installer
├── setup_venv.bat                 # Windows one-shot installer
│
├── .github/workflows/
│   └── release.yml                # CI: builds binaries for all 4 platforms on tag push
│
├── data/
│   ├── preprocessing.py           # TIF slicing, SHP burning, YOLO label gen
│   └── dataset.py                 # PyTorch Datasets + Albumentations pipelines
│
├── models/
│   ├── stage1_segmentation.py     # UNet++ + TriLoss + Lovász + TTA
│   └── stage2_models.py           # ConvNeXt-Large + ArcFace + YOLOv9 + SAHI
│
├── train/
│   ├── train_stage1.py            # SAM, EMA, SWA, grad checkpointing
│   ├── train_stage2.py            # Classifier + YOLO training loops
│   └── launch_ddp.py              # Multi-GPU DDP launcher
│
├── inference/
│   └── pipeline.py                # Batched multi-stage inference + shapefile export
│
├── utils/
│   ├── hardware.py                # Multi-backend setup, AMP, EMA, VRAM stats
│   ├── sam.py                     # Sharpness-Aware Minimisation optimiser
│   ├── ddp.py                     # DistributedDataParallel utilities
│   ├── metrics.py                 # mIoU, Dice, per-class IoU
│   ├── postprocess.py             # DenseCRF, watershed, morphology, vectorisation
│   ├── checkpointing.py           # Atomic checkpoint save
│   ├── logger.py                  # Structured logging + crash recovery
│   └── window.py                  # Cosine spline blending for tile stitching
│
└── tests/
    ├── test_config_values.py      # Config value assertions
    └── test_core_components.py    # Unit tests for core pipeline components
```

---

## Key Accuracy Features

| Feature | Stage | Impact |
|---------|-------|--------|
| UNet++ deep supervision | 1 | Sharper building boundaries |
| Lovász-Softmax loss | 1 | Directly optimises mIoU |
| Instance-touching separation loss | 1 | Prevents adjacent buildings merging |
| Watershed instance separation | 1 | Clean split of touching footprints |
| ArcFace angular-margin head | 2A | Better RCC vs Tiled discrimination |
| SAHI sliced inference | 2B | 640px slices — detects small wells at tile boundaries |
| CosineAnnealingWarmRestarts | 1 | Periodic LR re-exploration (works with SAM) |
| Road class weight 4.5× | 1 | Forces thin path connectivity |
| SAHI overlap 40% | 2B | Larger context for boundary objects |

---

## Configuration Reference

All hyperparameters live in `config.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `DEVICE` | auto | cuda → mps → cpu, auto-detected |
| `AMP_DTYPE` | auto | bf16 (CUDA) · fp16 (MPS) · fp32 (CPU) |
| `STAGE1["encoder"]` | `mit_b5` | 84M params |
| `STAGE1["batch_size"]` | `4` | Auto-halved when SAM + ≤16 GB VRAM |
| `STAGE1["class_weights"]` | `[0.30,1.80,4.50,2.20]` | Road 4.5× |
| `STAGE2A["arcface_m"]` | `0.55` | Angular margin |
| `STAGE2B["sahi_overlap_ratio"]` | `0.40` | Slice overlap |
| `STAGE2B["class_conf_thresh"]["well"]` | `0.03` | Low threshold for small objects |
| `STAGE2B["agnostic_nms"]` | `True` | Suppress cross-class duplicates |

---

## Deliverables

- **Building footprints** — Polygon shapefiles with watershed-separated instances
- **Rooftop classification** — `roof_type` attribute (RCC / Tiled / Tin / Other)
- **Road networks** — Contiguous polygon shapefiles
- **Waterbodies** — Polygon shapefiles
- **Infrastructure points** — Transformer / well / overhead tank point shapefiles

---

## License

Built for the NIF Hackathon.
