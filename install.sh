#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Geo-Intel Pipeline — Installer (macOS + Linux)
#  Detects: Apple Silicon / Intel Mac / NVIDIA CUDA / AMD ROCm / CPU
#
#  Usage:
#    chmod +x install.sh && ./install.sh
#
#  Options:
#    --cpu        Force CPU-only install (skip GPU detection)
#    --rocm VER   Force ROCm version  (e.g. --rocm 6.2)
#    --cuda VER   Force CUDA version  (e.g. --cuda 12.1)
#    --no-geo     Skip geospatial stack (rasterio/fiona/geopandas)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*"; exit 1; }
step()  { echo -e "\n${BOLD}$*${RESET}"; }

# ── Args ─────────────────────────────────────────────────────────────────────
FORCE_CPU=0; FORCE_ROCM=""; FORCE_CUDA=""; SKIP_GEO=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)      FORCE_CPU=1 ;;
    --rocm)     FORCE_ROCM="$2"; shift ;;
    --cuda)     FORCE_CUDA="$2"; shift ;;
    --no-geo)   SKIP_GEO=1 ;;
    *) warn "Unknown flag: $1" ;;
  esac
  shift
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║          Geo-Intel Pipeline — Installer                  ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Detect OS ────────────────────────────────────────────────────────────────
OS=""
ARCH=$(uname -m)
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
  OS="linux"
else
  error "Unsupported OS: $OSTYPE  (use setup_venv.bat on Windows)"
fi
info "Platform: $OS / $ARCH"

# ── Check Python ─────────────────────────────────────────────────────────────
step "1/7  Checking Python"
PYTHON=""
for cmd in python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=${VER%%.*}; MINOR=${VER##*.}
    if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]]; then
      PYTHON="$cmd"
      ok "$cmd $VER"
      break
    fi
  fi
done
[[ -z "$PYTHON" ]] && error "Python 3.10+ not found. Install from https://python.org"

# ── Create venv ───────────────────────────────────────────────────────────────
step "2/7  Setting up virtual environment"
VENV_DIR="$(pwd)/venv"
if [[ -d "$VENV_DIR" ]]; then
  info "venv already exists at $VENV_DIR"
else
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Created $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip upgraded"

# ── Detect accelerator ────────────────────────────────────────────────────────
step "3/7  Detecting GPU / accelerator"
TORCH_INDEX=""
TORCH_LABEL=""

if [[ "$FORCE_CPU" -eq 1 ]]; then
  TORCH_LABEL="CPU (forced)"

elif [[ -n "$FORCE_CUDA" ]]; then
  TAG="cu${FORCE_CUDA/./}"  # 12.1 → cu121
  TORCH_INDEX="https://download.pytorch.org/whl/${TAG}"
  TORCH_LABEL="NVIDIA CUDA $FORCE_CUDA (forced)"

elif [[ -n "$FORCE_ROCM" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/rocm${FORCE_ROCM}"
  TORCH_LABEL="AMD ROCm $FORCE_ROCM (forced)"

elif [[ "$OS" == "macos" ]]; then
  # Apple Silicon gets MPS via the standard wheel; Intel Mac gets CPU
  if [[ "$ARCH" == "arm64" ]]; then
    TORCH_LABEL="Apple Silicon (MPS via Metal)"
  else
    TORCH_LABEL="macOS Intel (CPU)"
  fi

elif [[ "$OS" == "linux" ]]; then
  # AMD ROCm check: presence of /dev/kfd or rocm-smi
  if [[ -e /dev/kfd ]] || command -v rocm-smi &>/dev/null; then
    ROC_VER=$(rocm-smi --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "6.2")
    TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROC_VER}"
    TORCH_LABEL="AMD ROCm $ROC_VER"
    # ROCm needs HSA runtime
    if [[ ! -e /dev/kfd ]]; then
      warn "/dev/kfd not found — ROCm GPU access may be limited"
    fi

  # NVIDIA CUDA check
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_RAW=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "")
    if [[ -z "$CUDA_RAW" ]]; then
      CUDA_RAW=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "12.1")
    fi
    CUDA_MAJOR=$(echo "$CUDA_RAW" | cut -d. -f1)
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      TORCH_LABEL="NVIDIA CUDA 12.x ($CUDA_RAW detected)"
    else
      TORCH_INDEX="https://download.pytorch.org/whl/cu118"
      TORCH_LABEL="NVIDIA CUDA 11.8 (detected $CUDA_RAW)"
    fi
  else
    TORCH_LABEL="CPU (no GPU detected)"
  fi
fi

info "Accelerator: $TORCH_LABEL"

# ── Install PyTorch ──────────────────────────────────────────────────────────
step "4/7  Installing PyTorch"
if [[ -n "$TORCH_INDEX" ]]; then
  pip install torch torchvision torchaudio \
    --index-url "$TORCH_INDEX" --quiet
else
  # CPU / Apple Silicon — standard index
  pip install torch torchvision torchaudio --quiet
fi

# Verify
python -c "
import torch, sys
dev = 'cuda' if torch.cuda.is_available() else \
      ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')
print(f'  torch {torch.__version__}  device={dev}')
if dev == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"
ok "PyTorch installed"

# ── Install geospatial stack ─────────────────────────────────────────────────
step "5/7  Installing geospatial stack"
if [[ "$SKIP_GEO" -eq 1 ]]; then
  warn "Skipping geospatial stack (--no-geo)"
else
  # conda path first (handles ECW + proper GDAL binary)
  if command -v conda &>/dev/null; then
    info "conda found — installing via conda-forge (includes ECW driver)"
    conda install -c conda-forge libgdal-ecw rasterio fiona geopandas \
      --yes --quiet 2>/dev/null || \
    conda install -c conda-forge rasterio fiona geopandas --yes --quiet
    ok "Geospatial stack installed via conda"
  else
    info "conda not found — using pip wheels"
    if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
      # arm64 wheels ship GDAL internally
      pip install rasterio fiona geopandas --quiet
    elif [[ "$OS" == "linux" ]]; then
      # Try system GDAL first, then pre-built wheels
      pip install rasterio fiona geopandas --quiet 2>/dev/null || \
      pip install rasterio fiona geopandas \
        --find-links https://github.com/cgohlke/geospatial-wheels/releases \
        --quiet
    else
      pip install rasterio fiona geopandas --quiet
    fi
    ok "Geospatial stack installed via pip"
  fi
fi

# ── Install ML + pipeline deps ───────────────────────────────────────────────
step "6/7  Installing ML & pipeline dependencies"
pip install \
  segmentation-models-pytorch>=0.3.3 \
  timm>=0.9.12 \
  ultralytics>=8.1.0 \
  shapely>=2.0.3 pyproj>=3.6.1 \
  albumentations>=1.4.0 "opencv-python>=4.9.0" "Pillow>=10.2.0" \
  scikit-image>=0.22.0 sahi>=0.11.15 \
  "numpy>=1.26.0" "pandas>=2.1.0" "scipy>=1.12.0" "scikit-learn>=1.4.0" \
  "tqdm>=4.66.1" pyyaml>=6.0.1 "matplotlib>=3.8.0" \
  "tensorboard>=2.16.0" psutil>=5.9.0 \
  "PyQt6>=6.6.0" lxml pytest>=8.0.0 \
  --quiet
ok "ML & pipeline deps installed"

# Optional: pydensecrf2 (C extension, may need build tools)
pip install pydensecrf2 --quiet 2>/dev/null \
  && ok "pydensecrf2 (CRF boundary refinement) installed" \
  || warn "pydensecrf2 not available — CRF post-processing will be skipped"

# ── Verify environment ────────────────────────────────────────────────────────
step "7/7  Verification"
python -c "
pkgs = [
    ('torch',                     lambda m: m.__version__),
    ('cv2',                       lambda m: m.__version__),
    ('numpy',                     lambda m: m.__version__),
    ('albumentations',            lambda m: m.__version__),
    ('segmentation_models_pytorch', lambda m: m.__version__),
    ('timm',                      lambda m: m.__version__),
    ('ultralytics',               lambda m: m.__version__),
    ('PyQt6',                     lambda m: m.QtCore.PYQT_VERSION_STR),
    ('matplotlib',                lambda m: m.__version__),
]
ok, fail = [], []
for name, ver_fn in pkgs:
    try:
        import importlib
        m = importlib.import_module(name)
        ok.append(f'{name:<36} {ver_fn(m)}')
    except Exception as e:
        fail.append(f'{name:<36} MISSING ({e})')
print()
for line in ok:   print(f'  ✓  {line}')
for line in fail: print(f'  ✗  {line}')
if fail:
    import sys; sys.exit(1)
"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Setup complete!${RESET}"
echo ""
echo "  Activate next time:"
echo "    source venv/bin/activate"
echo ""
echo "  Launch GUI:"
echo "    python gui.py"
echo ""
echo "  Preprocess your data:"
echo "    python run_pipeline.py --mode preprocess --data_root ./dataset"
echo ""
echo "  Train:"
echo "    python run_pipeline.py --mode train_all"
echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
echo ""
