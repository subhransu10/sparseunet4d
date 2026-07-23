# SparseUNet4D — venv deployment on a laptop (no conda)

Run the LiDAR motion-segmentation model on a laptop alongside an existing
**ROS 2 Humble + Husky** install, **without conda** and **without breaking**
the system Python or ROS. All ML dependencies live in one throwaway venv;
only a single, removable system package (`nvidia-cuda-toolkit`) is added.

Verified on: **Ubuntu 22.04**, **RTX 3050 Ti Laptop (4 GB VRAM, Ampere sm_86)**,
system **Python 3.10.12**, NVIDIA driver already installed, ROS 2 Humble present.

---

## What lives where (the isolation contract)

| Component | Location | Removal |
|---|---|---|
| torch, MinkowskiEngine, numpy, … | `~/mos_venv/` (venv) | `rm -rf ~/mos_venv` |
| `nvcc` + CUDA headers/libs (to *build* MinkowskiEngine) | `nvidia-cuda-toolkit` (system) | `sudo apt remove nvidia-cuda-toolkit` |
| `gcc-10` (host compiler nvcc needs) | system, *additive* next to gcc-11 | `sudo apt remove gcc-10 g++-10` |
| model code | `~/sparseunet4d/` | `rm -rf ~/sparseunet4d` |
| model weights `best.pt` | copied in by hand (not in git) | delete the file |

Nothing here overwrites a system library. ROS 2 and the Husky sim are never
touched — they keep importing from `/usr/lib/python3.10`, which the venv never
writes to.

---

## Prerequisites (already true on the target machine)

- NVIDIA driver working: `nvidia-smi` prints your GPU.
- ROS 2 Humble installed: `/opt/ros/humble/setup.bash` exists.
- System Python is 3.10 (`python3.10 --version`) — this is what ROS's `rclpy`
  is built against, so the venv **must** be created from it.

---

## Setup steps

### 1. System packages (additive, removable, do NOT touch ROS)

```bash
sudo apt update
sudo apt install -y python3.10-venv git wget gcc-10 g++-10 libopenblas-dev
sudo apt install -y nvidia-cuda-toolkit          # provides /usr/bin/nvcc
which nvcc && nvcc --version | tail -3            # must print a CUDA version
```

> `nvidia-cuda-toolkit` installs the CUDA **compiler** (`nvcc`), *not* a new
> GPU driver. It does not touch your existing driver or ROS.

### 2. Create the venv from system python3.10

```bash
python3.10 -m venv ~/mos_venv
source ~/mos_venv/bin/activate
pip install --upgrade pip wheel
```

### 3. PyTorch 1.12.1 + cu113 (canonical zero-patch MinkowskiEngine combo)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```
(~1.8 GB download; if your connection drops, pip resumes automatically.)

### 4. Build dependencies

```bash
pip install "numpy<2" pyyaml scipy ninja
```
`numpy<2` is required — torch 1.12 / MinkowskiEngine do not work with numpy 2.

### 5. Build MinkowskiEngine 0.5.4  (ONE job at a time — see notes!)

```bash
export CUDA_HOME=/usr
export PATH=/usr/bin:$PATH
export CC=gcc-10 CXX=g++-10
export TORCH_CUDA_ARCH_LIST="8.6"     # 3050 Ti = Ampere sm_86; use your GPU's arch
export MAX_JOBS=1                     # <-- CRITICAL on laptops, see notes

git clone https://github.com/NVIDIA/MinkowskiEngine.git ~/MinkowskiEngine
cd ~/MinkowskiEngine
python setup.py install --blas=openblas --force_cuda
```
Takes 10–20 min with `MAX_JOBS=1`. Long gaps between output lines are normal
(a big kernel compiling). Success ends with
`Finished processing dependencies for MinkowskiEngine==0.5.4`.

### 6. Model code + weights

```bash
git clone -b sota-push https://github.com/subhransu10/sparseunet4d ~/sparseunet4d
mkdir -p ~/sparseunet4d/runs/consistency_ft
# copy best.pt from wherever you trained it (scp or USB) into:
#   ~/sparseunet4d/runs/consistency_ft/best.pt
```

### 7. Verify the whole stack imports together

```bash
source /opt/ros/humble/setup.bash
source ~/mos_venv/bin/activate
python - <<'PY'
import torch, MinkowskiEngine as ME, rclpy
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("MinkowskiEngine:", ME.__version__)
print("rclpy OK")
PY
```
Expected: `cuda: True`, your GPU name, `MinkowskiEngine: 0.5.4`, `rclpy OK`.

---

## Every session, before running the node

```bash
source /opt/ros/humble/setup.bash
source ~/mos_venv/bin/activate
export SU4D_BACKEND=me          # REQUIRED: use MinkowskiEngine, not the CPU stand-in
```
Run the Husky sim in a **separate terminal with no venv activated** — the sim
talks to the node over ROS topics, not through shared Python.

> **`SU4D_BACKEND=me` is mandatory.** The code defaults to a pure-PyTorch "mock"
> backend (for laptops without MinkowskiEngine). A checkpoint trained with real
> ME will *not* load under the mock backend — every weight shows up as
> "missing" (`lin.weight` vs `conv.kernel` / `bn.bn.bn`). Setting this env var
> before Python starts selects the real backend and the checkpoint loads.

---

## Errors you may hit (and the fix)

**`ERROR: No matching distribution found for nvidia-cudart-cu11`**
That pip package name doesn't exist. Don't try to get `nvcc` from pip wheels —
use the system toolkit instead: `sudo apt install nvidia-cuda-toolkit` (step 1).

**`Command 'nvcc' not found` / `No such file or directory: '.../nvcc'`**
The CUDA toolkit isn't installed or isn't on `PATH`. Run
`sudo apt install -y nvidia-cuda-toolkit`, then `export CUDA_HOME=/usr` and
`export PATH=/usr/bin:$PATH` before building.

**The whole PC freezes during `python setup.py install`.**
The default build compiles all CUDA kernels in parallel, one per CPU core, and
each `nvcc` job needs 2–4 GB RAM — a laptop runs out of memory and locks up.
Fix: hard-reboot, then rebuild with **`export MAX_JOBS=1`** (step 5). Close
Chrome / VS Code / the sim first. If RAM is very tight (< 16 GB), add a swap
file for headroom:
```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

**`torch` download times out mid-way.**
pip resumes the partial download on its own; just re-run the `pip install torch`
line if it fully aborts.

**pip warns about `generate-parameter-library-py … requires jinja2 / typeguard`.**
Harmless. Those are *system ROS* packages seen because ROS was sourced; the
venv doesn't need them. Ignore.

**`numpy 2.x` gets installed by torchvision, then something breaks.**
Run `pip install "numpy<2"` to pin it back to 1.26 (step 4). This is expected.

**apt shows `NO_PUBKEY` / signature warnings for microsoft/edge repos.**
Unrelated third-party apt repos on the machine; they don't affect this setup.

**Out-of-memory (OOM) at *runtime* on 4 GB VRAM.**
A full 64-beam KITTI scan over 5 frames can exceed 4 GB. The Husky sim's LiDAR
(≈16 beams) is far lighter and should fit. If a live scan OOMs: reduce
`n_frames`, increase `voxel_size`, or run the node on a bigger GPU and keep only
the sim on the laptop. Skip the KITTI replay self-test on 4 GB — it's the
heaviest case and not representative of the sim load.

**Loading the checkpoint fails with every weight "missing" (`AssertionError`
listing `stem...`, `encoders...`, `decoders...`).**
You forgot `export SU4D_BACKEND=me`. The model built the pure-PyTorch mock
backend (`lin.weight`) but the checkpoint holds MinkowskiEngine weights
(`conv.kernel`, `bn.bn.bn`). Set the env var **before** running Python.

**GitHub shows the commits as "Unverified".**
Cosmetic (missing commit signature), unrelated to the code. Set up SSH commit
signing on your dev machine if you want the green badge.

---

## Notes on versions

- `nvcc` from `nvidia-cuda-toolkit` on 22.04 is ~CUDA 11.5, while torch is
  built with cu113. A version-mismatch **warning** during the build is
  harmless — the CUDA 11.x ABI is compatible and MinkowskiEngine links fine.
- Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (3050 Ti = `8.6`;
  find yours with `python -c "import torch; print(torch.cuda.get_device_capability())"`).
