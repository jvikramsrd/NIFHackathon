#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Geo-Intel Pipeline — Installer (macOS / Linux)
#
#  Auto-detects & selects the correct PyTorch requirements file:
#    Apple Silicon MPS    →  requirements-torch-cpu.txt
#    macOS Intel          →  requirements-torch-cpu.txt
#    NVIDIA CUDA 12.x     →  requirements-torch-cuda.txt   (cu121)
#    NVIDIA CUDA 11.x     →  requirements-torch-cuda11.txt  (cu118)
#    AMD ROCm 6.x         →  requirements-torch-rocm.txt   (rocm6.2)
#    AMD ROCm 5.x         →  direct --index-url            (no static file)
#    CPU / no GPU         →  requirements-torch-cpu.txt
#
#  Options:
#    --cpu        Force CPU-only install
#    --cuda VER   Force CUDA version  (e.g. --cuda 12.1)
#    --rocm VER   Force ROCm version  (e.g. --rocm 6.2)
#    --no-geo     Skip geospatial stack (rasterio/fiona/geopandas)
#    --no-crf     Skip pydensecrf2 (CRF post-processing)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*"; exit 1; }
step()  { echo -e "\n  ${BOLD}$*${RESET}"; }

# ── Parse args ───────────────────────────────────────────────────────────────
FORCE_CPU=0; FORCE_ROCM=""; FORCE_CUDA=""; SKIP_GEO=0; SKIP_CRF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)    FORCE_CPU=1 ;;
    --rocm)   FORCE_ROCM="$2"; shift ;;
    --cuda)   FORCE_CUDA="$2"; shift ;;
    --no-geo) SKIP_GEO=1 ;;
    --no-crf) SKIP_CRF=1 ;;
    *) warn "Unknown flag: $1" ;;
  esac
  shift
done

echo ""
echo -e "  ${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${BOLD}║       Geo-Intel Pipeline — Installer                 ║${RESET}"
echo -e "  ${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Detect OS ───────────────────────────────────────────────────────────────
OS=""; ARCH=$(uname -m)
if [[ "$OSTYPE" == "darwin"* ]]; then OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then OS="linux"
else error "Unsupported OS: $OSTYPE  (use setup_venv.bat on Windows)"
fi
info "Platform: ${OS} / ${ARCH}"

# ── 1/6  Check Python ───────────────────────────────────────────────────────
step "1/6  Checking Python"
PYTHON=""
for cmd in python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=${VER%%.*}; MINOR=${VER##*.}
    if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]]; then
      PYTHON="$cmd"; ok "$cmd $VER"; break
    fi
  fi
done
[[ -z "$PYTHON" ]] && error "Python 3.10+ not found.  https://python.org"

# ── 2/6  Virtual environment ────────────────────────────────────────────────
step "2/6  Virtual environment"
VENV_DIR="$(pwd)/venv"
if [[ -d "$VENV_DIR" ]]; then
  info "venv already exists"
else
  "$PYTHON" -m venv "$VENV_DIR"; ok "created $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip upgraded"

# ── 3/6  Accelerator detection — select the right torch requirements file ───
step "3/6  Detecting accelerator"
TORCH_FILE=""; TORCH_URL=""; TORCH_LABEL=""

if [[ "$FORCE_CPU" -eq 1 ]]; then
  TORCH_FILE="requirements-torch-cpu.txt"; TORCH_LABEL="CPU (forced)"

elif [[ -n "$FORCE_CUDA" ]]; then
  if [[ "${FORCE_CUDA%%.*}" -ge 12 ]]; then TORCH_FILE="requirements-torch-cuda.txt"
  else TORCH_FILE="requirements-torch-cuda11.txt"; fi
  TORCH_LABEL="NVIDIA CUDA $FORCE_CUDA (forced)"

elif [[ -n "$FORCE_ROCM" ]]; then
  if [[ "${FORCE_ROCM%%.*}" -ge 6 ]]; then TORCH_FILE="requirements-torch-rocm.txt"
  else TORCH_URL="https://download.pytorch.org/whl/rocm${FORCE_ROCM}"; fi
  TORCH_LABEL="AMD ROCm $FORCE_ROCM (forced)"

elif [[ "$OS" == "macos" ]]; then
  TORCH_FILE="requirements-torch-cpu.txt"
  [[ "$ARCH" == "arm64" ]] && TORCH_LABEL="Apple Silicon (MPS)" || TORCH_LABEL="macOS Intel (CPU)"

elif [[ "$OS" == "linux" ]]; then
  # AMD ROCm
  if [[ -e /dev/kfd ]] || command -v rocm-smi &>/dev/null; then
    ROC_VER=$(rocm-smi --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || true)
    [[ -z "$ROC_VER" ]] && ROC_VER="6.2"
    if [[ "${ROC_VER%%.*}" -ge 6 ]]; then TORCH_FILE="requirements-torch-rocm.txt"
    else TORCH_URL="https://download.pytorch.org/whl/rocm${ROC_VER}"; fi
    TORCH_LABEL="AMD ROCm $ROC_VER (detected)"
    [[ ! -e /dev/kfd ]] && warn "/dev/kfd not found — GPU access may be limited"
  # NVIDIA CUDA
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_RAW=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "")
    [[ -z "$CUDA_RAW" ]] && CUDA_RAW=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "12.1")
    if [[ "${CUDA_RAW%%.*}" -ge 12 ]]; then TORCH_FILE="requirements-torch-cuda.txt"
    else TORCH_FILE="requirements-torch-cuda11.txt"; fi
    TORCH_LABEL="NVIDIA CUDA $CUDA_RAW (detected)"
  else
    TORCH_FILE="requirements-torch-cpu.txt"; TORCH_LABEL="CPU (no GPU detected)"
  fi
fi

if [[ -n "$TORCH_FILE" ]]; then
  ok "$TORCH_LABEL  →  $TORCH_FILE"
else
  ok "$TORCH_LABEL  →  $TORCH_URL"
fi

# ── 4/6  Install PyTorch ────────────────────────────────────────────────────
step "4/6  Installing PyTorch"
if [[ -n "${TORCH_FILE:-}" ]]; then
  pip install -r "$TORCH_FILE" --quiet
else
  pip install torch torchvision torchaudio --index-url "$TORCH_URL" --quiet
fi

python -c "
import torch
dev = 'cuda' if torch.cuda.is_available() else \
      ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
print(f'  torch {torch.__version__}  device={dev}')
if dev == 'cuda':
    print(f'  GPU:  {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"
ok "PyTorch ready"

# ── 5/6  Geospatial stack ───────────────────────────────────────────────────
step "5/6  Installing geospatial stack"
if [[ "$SKIP_GEO" -eq 1 ]]; then
  warn "Skipped (--no-geo)"
else
  if command -v conda &>/dev/null; then
    info "conda found — installing with ECW support"
    conda install -c conda-forge libgdal-ecw rasterio fiona geopandas --yes --quiet 2>/dev/null || \
      conda install -c conda-forge rasterio fiona geopandas --yes --quiet
    ok "Installed via conda"
  else
    info "conda not found — using pip"
    if [[ "$OS" == "linux" ]]; then
      pip install rasterio fiona geopandas --quiet 2>/dev/null || \
        pip install rasterio fiona geopandas --find-links https://github.com/cgohlke/geospatial-wheels/releases --quiet
    else
      pip install rasterio fiona geopandas --quiet
    fi
    ok "Installed via pip"
  fi
fi

# ── 6/6  Pipeline dependencies + CRF + verify ───────────────────────────────
step "6/6  Installing pipeline dependencies (requirements.txt)"
pip install -r requirements.txt --quiet 2>/dev/null || {
  warn "onnxruntime-gpu unavailable — retrying with CPU onnxruntime"
  TMP_REQ="$(mktemp)"
  grep -v 'onnxruntime' requirements.txt > "$TMP_REQ"
  pip install -r "$TMP_REQ" --quiet
  pip install onnxruntime --quiet 2>/dev/null || true
  rm -f "$TMP_REQ"
}
ok "Dependencies installed"

if [[ "$SKIP_CRF" -eq 0 ]]; then
  pip install pydensecrf2 --quiet 2>/dev/null && ok "pydensecrf2 (CRF)" || warn "pydensecrf2 skipped — CRF disabled"
fi

# Check ECW driver
echo ""
python -c "
try:
    from osgeo import gdal
    n = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
    print(f'  ECW driver: {\"YES\" if \"ECW\" in n else \"NO (conda install -c conda-forge libgdal-ecw)\"}')
except: pass
" 2>/dev/null || true

# ── Verify ──────────────────────────────────────────────────────────────────
python -c "
import importlib, torch
ok, fail = [], []

try:
    import torch; ok.append(('torch', torch.__version__))
    d = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
    ok.append(('device', d))
    if d == 'cuda':
        ok.append(('GPU', torch.cuda.get_device_name(0)))
        ok.append(('VRAM', f\"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\"))
except Exception as e: fail.append(('torch', str(e)))

for pkg, attr in [('timm','__version__'),('ultralytics','__version__'),
                   ('segmentation_models_pytorch','__version__'),
                   ('cv2','__version__'),('albumentations','__version__')]:
    try:
        m = importlib.import_module(pkg); ok.append((pkg, getattr(m, attr)))
    except Exception as e: fail.append((pkg, str(e)))

for pkg in ['rasterio','geopandas','shapely','pyproj']:
    try:
        m = importlib.import_module(pkg); ok.append((pkg, m.__version__))
    except Exception as e: fail.append((pkg, str(e)))

for pkg, attr in [('numpy','__version__'),('pandas','__version__'),
                   ('PyQt6','QtCore.PYQT_VERSION_STR'),('matplotlib','__version__'),
                   ('tensorboard','__version__'),('onnx','__version__')]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, attr, None)
        if callable(v): v = v()
        ok.append((pkg, v or getattr(m, '__version__', '?')))
    except Exception as e: fail.append((pkg, str(e)))

try:
    import wandb; ok.append(('wandb', wandb.__version__))
except: pass

print()
for k, v in ok:   print(f'  \u2713  {k:<34} {v}')
for k, v in fail: print(f'  \u2717  {k:<34} MISSING ({v})')
if fail: import sys; sys.exit(1)
"

echo ""
echo -e "  ${BOLD}══════════════════════════════════════════════════════${RESET}"
echo -e "  ${GREEN}Setup complete!${RESET}"
echo ""
echo "    Activate next time:  source venv/bin/activate"
echo "    Launch GUI:          python gui.py"
echo "    Preprocess:          python run_pipeline.py --mode preprocess --data_root ./dataset"
echo "    Train:               python run_pipeline.py --mode train_all"
echo -e "  ${BOLD}══════════════════════════════════════════════════════${RESET}"
echo ""