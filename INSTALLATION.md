# Installation

This guide starts from a fresh Ubuntu 22.04 machine with an NVIDIA GPU. It keeps
all Python packages in `~/mos_venv`; it does not replace the system Python or
ROS installation.

## 1. Prerequisites

Verify the NVIDIA driver and Python version:

```bash
nvidia-smi
python3.10 --version
```

The tested software stack is:

| Component | Version |
|---|---|
| Ubuntu | 22.04 |
| Python | 3.10.12 |
| PyTorch | 1.12.1+cu113 |
| torchvision | 0.13.1+cu113 |
| MinkowskiEngine | 0.5.4 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| PyYAML | 6.0.3 |

A newer NVIDIA driver is fine: the driver-reported CUDA version does not need
to equal PyTorch's bundled CUDA 11.3 runtime.

## 2. Clone the repository

```bash
git clone https://github.com/subhransu10/sparseunet4d.git
cd sparseunet4d
```

## 3. Automated installation

The setup script installs build prerequisites, creates the virtual environment,
installs the pinned PyTorch build, and compiles MinkowskiEngine for the local
GPU. Compilation is intentionally limited to one job to avoid exhausting RAM.

```bash
bash deploy/setup_venv.sh
source ~/activate_mos.sh
```

The script is safe to rerun after a failed or interrupted build.

## 4. Download the model

```bash
cd ~/sparseunet4d
bash deploy/download_model.sh
```

This downloads and verifies:

```text
checkpoints/sparseunet4d_semantickitti/best.pt
SHA-256: 65f7525f00a4a490df30ec91b5db713d865f30dffd76b4c7f9dfcbc353e31f1c
```

If downloading manually from the GitHub Releases page, create the directory and
place the file at that exact path:

```bash
mkdir -p checkpoints/sparseunet4d_semantickitti
mv ~/Downloads/sparseunet4d_semantickitti_best.pt \
  checkpoints/sparseunet4d_semantickitti/best.pt
sha256sum checkpoints/sparseunet4d_semantickitti/best.pt
```

## 5. Verify the environment and checkpoint

```bash
source ~/activate_mos.sh
cd ~/sparseunet4d

python - <<'PY'
import torch
import MinkowskiEngine as ME
from sparseunet4d.models.backend import backend
from mos_inference import MOSInference

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("MinkowskiEngine:", ME.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Backend:", backend())

model = MOSInference(
    "configs/pretrained_semantickitti.yaml",
    "checkpoints/sparseunet4d_semantickitti/best.pt",
    device="cuda",
)
print("Checkpoint threshold:", model.threshold)
assert torch.cuda.is_available()
assert backend() == "me"
PY
```

Expected output includes MinkowskiEngine `0.5.4`, backend `me`, and checkpoint
threshold `0.9`.

## Troubleshooting

- **The PC freezes while compiling MinkowskiEngine:** close memory-heavy apps,
  confirm `MAX_JOBS=1`, and rerun the setup script. On a 16 GB computer, an
  8 GB swapfile is advisable during compilation.
- **Checkpoint keys are reported missing:** `SU4D_BACKEND=me` was not exported.
  Run `source ~/activate_mos.sh` in the current terminal.
- **CUDA out of memory at runtime:** close GPU applications, use the deployment
  config with a shorter range, or run on a GPU with more VRAM.
- **`nvidia-smi` fails:** repair the NVIDIA driver first; this repository does
  not install or replace it.
- **MinkowskiEngine was compiled for another GPU:** rebuild it locally with the
  correct `TORCH_CUDA_ARCH_LIST`.

Continue with [DEPLOYMENT.md](DEPLOYMENT.md).
