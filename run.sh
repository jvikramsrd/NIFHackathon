#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Geo-Intel Pipeline — CLI wrapper (macOS / Linux)
#
#  Resolution order:
#    1. If dist/geo-intel/geo-intel exists (a built binary), use it.
#    2. Else activate venv (if any) and invoke run_pipeline.py.
#
#  Usage:
#    ./run.sh --mode preprocess --data_root ./dataset
#    ./run.sh --mode train_all
#    ./run.sh --mode infer --tif ./village.tif --out ./outputs/village
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")"

# ── 1. Prefer the built CLI binary if present ────────────────────────────────
if [[ -x "dist/geo-intel/geo-intel" ]]; then
    exec "dist/geo-intel/geo-intel" "$@"
fi

# ── 2. Fallback: activate venv and invoke run_pipeline.py ────────────────────
if [[ -f "venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
else
    echo "[warn] No built binary in dist/ and no venv found." >&2
    echo "       Run \`./install.sh\` once to provision the Python environment," >&2
    echo "       or \`python build.py\` to produce dist/geo-intel/geo-intel." >&2
fi

exec python run_pipeline.py "$@"
