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
#    --help       Show this help
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n  ${BOLD}$*${RESET}"; }

usage() {
    grep '^#  ' "$0" | sed 's/^#  //'
    exit 0
}

# ── Parse args ───────────────────────────────────────────────────────────────
FORCE_CPU=0; FORCE_ROCM=""; FORCE_CUDA=""; SKIP_GEO=0; SKIP_CRF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage ;;
    --cpu)    FORCE_CPU=1 ;;
    --rocm)
      [[ $# -lt 2 || -z "${2:-}" ]] && error "--rocm requires a version (e.g. --rocm 6.2)"
      FORCE_ROCM="$2"; shift ;;
    --cuda)
      [[ $# -lt 2 || -z "${2:-}" ]] && error "--cuda requires a version (e.g. --cuda 12.1)"
      FORCE_CUDA="$2"; shift ;;
    --no-geo) SKIP_GEO=1 ;;
    --no-crf) SKIP_CRF=1 ;;
    *) error "Unknown flag: $1  (run with --help)" ;;
  esac
  shift
done

echo ""
echo -e "  ${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${BOLD}║       Geo-Intel Pipeline — Installer                 ║${RESET}"
echo -e "  ${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Validate required files ─────────────────────────────────────────────────
[[ -f requirements.txt ]] || error "requirements.txt not found — run this script from the project root"

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
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
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
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools>=69,<82" wheel --quiet
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
    ROC_VER=$(rocm-smi --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    [[ -z "$ROC_VER" ]] && ROC_VER="6.2"
    if [[ "${ROC_VER%%.*}" -ge 6 ]]; then TORCH_FILE="requirements-torch-rocm.txt"
    else TORCH_URL="https://download.pytorch.org/whl/rocm${ROC_VER}"; fi
    TORCH_LABEL="AMD ROCm $ROC_VER (detected)"
    [[ ! -e /dev/kfd ]] && warn "/dev/kfd not found — GPU access may be limited"
  # NVIDIA CUDA
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_RAW=$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    if [[ -z "$CUDA_RAW" ]]; then
      CUDA_RAW=$(nvcc --version 2>/dev/null | grep 'release' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    fi
    if [[ -z "$CUDA_RAW" ]]; then
      warn "nvidia-smi found but CUDA version detection failed — assuming CUDA 12.x"
      CUDA_RAW="12.1"
    fi
    if [[ "${CUDA_RAW%%.*}" -ge 12 ]]; then TORCH_FILE="requirements-torch-cuda.txt"
    else TORCH_FILE="requirements-torch-cuda11.txt"; fi
    TORCH_LABEL="NVIDIA CUDA $CUDA_RAW (detected)"
  else
    TORCH_FILE="requirements-torch-cpu.txt"; TORCH_LABEL="CPU (no GPU detected)"
  fi
fi

# Validate selected requirements file exists
if [[ -n "${TORCH_FILE:-}" ]]; then
  [[ -f "$TORCH_FILE" ]] || error "Requirements file not found: $TORCH_FILE"
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

python _setup_verify.py --torch
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

# Temp file for onnxruntime-gpu fallback — cleaned up on exit
TMP_REQ=""
cleanup_tmp() { [[ -n "${TMP_REQ:-}" ]] && rm -f "$TMP_REQ"; }
trap cleanup_tmp EXIT

pip install -r requirements.txt --quiet 2>/dev/null || {
  warn "onnxruntime-gpu unavailable — retrying with CPU onnxruntime"
  TMP_REQ="$(mktemp)"
  grep -v 'onnxruntime' requirements.txt > "$TMP_REQ"
  pip install -r "$TMP_REQ" --quiet
  pip install onnxruntime --quiet 2>/dev/null || true
}
ok "Dependencies installed"

if [[ "$SKIP_CRF" -eq 0 ]]; then
  pip install pydensecrf2 --quiet 2>/dev/null && ok "pydensecrf2 (CRF)" || warn "pydensecrf2 skipped — CRF disabled"
fi

# Check ECW driver
echo ""
python _setup_verify.py --ecw 2>/dev/null || true

# ── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}══════════════════════════════════════════════════════${RESET}"
echo -e "  ${BOLD}Verification${RESET}"
echo -e "  ${BOLD}══════════════════════════════════════════════════════${RESET}"
python _setup_verify.py --verify

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
