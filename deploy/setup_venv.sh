#!/usr/bin/env bash
# =============================================================================
#  SparseUNet4D — one-shot deployment setup (venv, no conda)
#
#  For an Ubuntu 22.04 machine with an NVIDIA GPU. Everything Python lives in
#  ~/mos_venv; the system additions are build tools and the CUDA toolkit.
#  ROS 2 Humble is detected and sourced when installed, but is not required for
#  offline inference or benchmarking.
#
#  Usage (from inside the cloned repo):
#     bash deploy/setup_venv.sh
#  Then each session:
#     source ~/activate_mos.sh
#
#  Idempotent: safe to re-run; it skips steps already done.
# =============================================================================
set -euo pipefail

VENV="$HOME/mos_venv"
ME_DIR="$HOME/MinkowskiEngine"

# locate the repo root (this script may sit in <repo>/deploy or <repo>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if   [ -f "$SCRIPT_DIR/mos_node.py" ];      then REPO="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../mos_node.py" ];   then REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
else echo "ERROR: run this from inside the sparseunet4d repo"; exit 1; fi
echo "repo: $REPO"

echo "==> [1/6] system packages (additive, removable; do NOT touch ROS/driver)"
sudo apt-get update
sudo apt-get install -y python3.10-venv git curl wget \
                        gcc-10 g++-10 libopenblas-dev nvidia-cuda-toolkit
command -v nvcc >/dev/null || { echo "ERROR: nvcc missing after install"; exit 1; }
echo "    nvcc: $(nvcc --version | tail -1)"

echo "==> [2/6] python venv from system python3.10"
[ -d "$VENV" ] || python3.10 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "==> [3/6] PyTorch 1.12.1 + cu113  (canonical zero-patch ME combo)"
python -c "import torch" 2>/dev/null || \
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
      --extra-index-url https://download.pytorch.org/whl/cu113

echo "==> [4/6] runtime deps"
pip install "numpy<2" pyyaml scipy ninja
pip install kiss-icp || echo "    (kiss-icp optional; skipped)"

echo "==> [5/6] build MinkowskiEngine 0.5.4 (ONE job -> no laptop OOM freeze)"
if python -c "import MinkowskiEngine" 2>/dev/null; then
  echo "    already installed, skipping build"
else
  export CUDA_HOME=/usr PATH="/usr/bin:$PATH" CC=gcc-10 CXX=g++-10 MAX_JOBS=1
  # auto-detect this GPU's compute capability (fallback 8.6)
  ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo 8.6)
  export TORCH_CUDA_ARCH_LIST="$ARCH"
  echo "    building for sm_${ARCH/./}"
  if [ ! -d "$ME_DIR/.git" ]; then
    git clone https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
  fi
  ( cd "$ME_DIR" && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --blas=openblas --force_cuda )
fi

echo "==> [6/6] write ~/activate_mos.sh (source this every session)"
cat > "$HOME/activate_mos.sh" <<EOF
# source this before running the MOS node / scripts
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
source "$VENV/bin/activate"
export SU4D_BACKEND=me          # REQUIRED: use MinkowskiEngine, not the CPU stub
export SU4D_REPO="$REPO"
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
EOF
echo "    created $HOME/activate_mos.sh"

echo "==> fetching pretrained checkpoint from GitHub release"
bash "$SCRIPT_DIR/download_model.sh" || \
  echo "    (download failed; fetch it manually — see deploy/DEPLOY_VENV.md)"

echo
echo "==> verify"
# shellcheck disable=SC1091
SU4D_BACKEND=me python - <<'PY'
import torch, MinkowskiEngine as ME
print("  cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("  MinkowskiEngine:", ME.__version__)
PY

echo
echo "======================================================================="
echo " DONE.  Next:"
echo "   1) every session:        source ~/activate_mos.sh"
echo "   2) see DEPLOYMENT.md for KITTI / Husky-sim / real-robot runs"
echo "   (checkpoint downloaded to checkpoints/sparseunet4d_semantickitti/best.pt)"
echo "======================================================================="
