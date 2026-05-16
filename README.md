# Geo-Intel — Production Geospatial CV Pipeline

> AI-based feature extraction from drone orthophotos, tuned for the SVAMITVA dataset.  
> Supports NVIDIA CUDA · AMD ROCm · Apple Silicon MPS · CPU.

---

## Pipeline Architecture

```
INPUT: SVAMITVA drone orthophoto (GeoTIFF / ECW) + Annotation Shapefiles
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Semantic Segmentation                                     │
│  Architecture : Unet + scSE attention                                │
│  Encoder      : MixTransformer B4 (mit_b4) — faster production fit  │
│  Input        : 512×512 patches, multi-scale (256/512/768)           │
│  Output       : 4-class mask → Background / Building / Road / Water  │
│  Loss         : TriLoss [Lovász-Softmax + Dice + Focal + Boundary +  │
│                 Instance-Touching Separation]                        │
│  Inference    : 3-scale TTA (0.875×, 1.0×, 1.25×) × D4 symmetry     │
│  Post-process : Watershed separation + DenseCRF + Morphology         │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
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

**The binary is only the GUI shell.** You still need a Python environment with PyTorch + pipeline deps installed (see Option B below) for training and inference. The GUI auto-detects your active Python.

```
# After unzipping:
Windows:    GeoIntel\GeoIntel.exe
```

---

### Option B — Install from source (for training / development)

#### Prerequisites

- Python 3.10 – 3.12 (3.13 has known PyQt6 DLL issues on Windows; use 3.12)
- Git

#### 1. Clone

```bash
git clone https://github.com/jvikramsrd/NIFHackathon.git
cd NIFHackathon
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
pip install rasterio fiona geopandas
```

**ECW support** (optional — needed only if your input imagery is `.ecw`):

ECW is a proprietary Hexagon format that requires a special GDAL build. The pipeline auto-converts ECW → GeoTIFF at runtime using any of these backends (tried in order):

1. **OSGeo4W** *(recommended on Windows, free)*
   - Download: https://trac.osgeo.org/osgeo4w/ → Express Install → select **GDAL**
   - The pipeline finds `C:\OSGeo4W\bin\gdal_translate.exe` automatically — no extra config needed

2. **QGIS** — if already installed, its bundled `gdal_translate.exe` is used automatically

3. **Native rasterio ECW driver** — works if your GDAL was compiled with the Hexagon SDK
   (not available via standard pip/conda channels as of 2025 due to license restrictions)

> `conda install -c conda-forge libgdal-ecw` will fail — this package was removed from conda-forge
> because Hexagon's SDK cannot be freely redistributed. OSGeo4W is the easiest workaround.

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

**Windows** — auto-detects NVIDIA CUDA 11/12, prompts for manual ROCm:
```
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
| 1 — Segmentation | Unet mit_b4 | 512×512 | 8 | ~12 GB | ~16 GB unified |
| 2A — Classification | ConvNeXt-Large | 224×224 | 32 | ~7.8 GB | ~8 GB unified |
| 2B — Detection | YOLOv9e | 1280×1280 | 2 | ~7.2 GB | ~9 GB unified |

---

## Quick Start

### Windows launchers (double-click)

| File | Action |
|------|--------|
| `launch_gui.bat` | Open the Operator Console GUI |
| `run.bat --mode ...` | CLI pipeline runner |

### Desktop GUI
```bash
python gui.py
# or on Windows:
launch_gui.bat
```

Tabs:
- **Pipeline Runner** — preprocess, train, evaluate, infer with live log + progress
- **Map Viewer** — side-by-side TIF vs. segmentation overlay with opacity slider
- **Results** — metrics table + bar chart, auto-reads `outputs/results.json`

### CLI

```bash
# Preprocess — strips rasters, burns masks, generates crops + YOLO labels
python run_pipeline.py --mode preprocess --data_root ./dataset

# Train all stages
python run_pipeline.py --mode train_all

# Evaluate (writes outputs/results.json)
python run_pipeline.py --mode evaluate

# Inference on a single image (.tif, .tiff, or .ecw)
python run_pipeline.py --mode infer \
  --tif "path/to/village.tif" \
  --out ./outputs/village_name

# Batch inference on a folder (all .tif / .tiff / .ecw files)
python infer_folder.py --test_folder "path/to/folder" --out_folder ./outputs/batch

# End-to-end
python run_pipeline.py --mode all --data_root ./dataset
```

On Windows, replace `python run_pipeline.py` with `run.bat`:
```
run.bat --mode infer --tif "path\to\village.ecw" --out .\outputs\village
```

> ECW files are auto-converted to GeoTIFF at runtime (requires OSGeo4W or QGIS — see ECW support above).

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
│   ├── village1.tif        ← GeoTIFF  ┐
│   ├── village1.ecw        ← ECW      ┘ either format accepted
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
NIFHackathon/
├── config.py                      # Hardware, paths, hyperparameters, class maps
├── run_pipeline.py                # Master CLI (preprocess/train/evaluate/infer)
├── gui.py                         # Desktop operator console (PyQt6)
├── infer_folder.py                # Standalone batch inference script
├── export_models.py               # ONNX export utility
├── build.py                       # Binary build script (PyInstaller)
├── geo_intel.spec                 # PyInstaller spec
├── pyproject.toml                 # pip-installable package definition
├── requirements.txt               # Base deps (no torch — platform-specific)
├── requirements-torch-cuda.txt    # NVIDIA CUDA 12.x torch
├── requirements-torch-cuda11.txt  # NVIDIA CUDA 11.x torch
├── requirements-torch-rocm.txt    # AMD ROCm torch
├── requirements-torch-cpu.txt     # CPU / Apple MPS torch
├── install.sh                     # macOS + Linux one-shot installer
├── setup_venv.bat                 # Windows one-shot setup
├── launch_gui.bat                 # Windows — double-click to launch GUI
├── run.bat                        # Windows — CLI wrapper for run_pipeline.py
├── PROJECT_REFERENCE.md           # Full technical reference (all components)
│
├── data/
│   ├── preprocessing.py           # TIF slicing, SHP burning, YOLO label gen
│   └── dataset.py                 # PyTorch Datasets + Albumentations pipelines
│
├── models/
│   ├── stage1_segmentation.py     # Unet + scSE + TriLoss + Lovász + TTA
│   └── stage2_models.py           # ConvNeXt-Large + ArcFace + YOLOv9 + SAHI
│
├── train/
│   ├── train_stage1.py            # EMA, SWA, grad checkpointing (SAM available, off by default)
│   └── train_stage2.py            # Classifier + YOLO training loops
│
├── inference/
│   └── pipeline.py                # Batched multi-stage inference + shapefile export
│
├── utils/
│   ├── hardware.py                # Multi-backend setup, AMP, EMA, VRAM stats
│   ├── sam.py                     # Sharpness-Aware Minimisation optimiser
│   ├── metrics.py                 # mIoU, Dice, per-class IoU
│   ├── postprocess.py             # DenseCRF, watershed, morphology, vectorisation
│   ├── checkpointing.py           # Atomic checkpoint save/load
│   ├── logger.py                  # Structured logging + crash recovery
│   ├── window.py                  # Cosine spline blending for tile stitching
│   └── ecw_compat.py              # ECW → GeoTIFF auto-conversion (rasterio / gdal_translate / osgeo)
│
└── tests/
    ├── test_config_values.py      # Config value assertions
    └── test_core_components.py    # Unit tests for core pipeline components
```

---

## Key Accuracy Features

| Feature | Stage | Impact |
|---------|-------|--------|
| Unet + scSE decoder attention | 1 | Channel + spatial recalibration on skip features |
| Lovász-Softmax loss | 1 | Directly optimises mIoU |
| Instance-touching separation loss | 1 | Prevents adjacent buildings merging |
| Watershed instance separation | 1 | Clean split of touching footprints |
| Percentile normalization (2nd–98th) | 1 | Robust to satellite radiometric outliers |
| Full 192 px tile overlap | 1 | Eliminates border blending artefacts |
| CRF 10 iterations | 1 | Boundary convergence (5 was insufficient) |
| min_fg_ratio 0.01 | 1 train | Filters near-empty training patches |
| ArcFace angular-margin head | 2A | Better RCC vs Tiled discrimination |
| ArcFace m=0.55 (from config) | 2A | Correct margin — was silently hardcoded 0.50 |
| Per-class conf thresholds (Stage 2A) | 2A | RCC=0.45 / Tiled=0.55 / Tin=0.50 / Other=0.40 |
| SAHI sliced inference | 2B | 640 px slices — detects small wells at tile edges |
| SAHI overlap 40% | 2B | Larger context for boundary objects |
| Soft-NMS σ=0.5 | 2B | Proper box suppression (was 0.9 — near-disabled) |
| Well conf threshold 0.10 | 2B | Eliminates false-positive well flood (was 0.03) |
| FAST_TTA flag | 1 | Toggle: 8-pass fast vs 24-pass accurate TTA |

---

## Configuration Reference

All hyperparameters live in `config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `DEVICE` | auto | cuda → mps → cpu, auto-detected |
| `AMP_DTYPE` | auto | bf16 (CUDA) · fp16 (MPS) · fp32 (CPU) |
| `FAST_TTA` | `False` | True = 8 passes, False = 24 passes (more accurate) |
| `STAGE1["arch"]` | `Unet` | smp decoder; MiT encoders are not compatible with `UnetPlusPlus` |
| `STAGE1["encoder"]` | `mit_b4` | Production speed/accuracy balance |
| `STAGE1["patch_size"]` | `512` | Training patch size |
| `STAGE1["overlap"]` | `192` | Tile overlap for seamless stitching |
| `STAGE1["batch_size"]` | `8` | ×4 grad accum = effective batch 32 (SAM off) |
| `STAGE1["epochs"]` | `80` | Default fine-tuning length for pretrained mit_b4 |
| `STAGE1["use_sam"]` | `False` | Enabling doubles per-iter cost and forces grad_accum=1 |
| `STAGE1["crf_iter"]` | `10` | CRF iterations at inference |
| `STAGE1["class_weights"]` | `[0.30,1.80,4.50,2.20]` | Road 4.5× to force thin path connectivity |
| `STAGE1["min_fg_ratio"]` | `0.01` | Drop training patches with <1% foreground |
| `STAGE2A["arcface_m"]` | `0.55` | Angular margin for rooftop classification |
| `STAGE2A["stage2a_conf_thresh"]` | per-class dict | RCC=0.45, Tiled=0.55, Tin=0.50, Other=0.40 |
| `STAGE2B["sahi_overlap_ratio"]` | `0.40` | Slice overlap for SAHI inference |
| `STAGE2B["soft_nms_sigma"]` | `0.5` | Gaussian NMS decay factor |
| `STAGE2B["class_conf_thresh"]["well"]` | `0.10` | Minimum confidence to emit a well detection |
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
