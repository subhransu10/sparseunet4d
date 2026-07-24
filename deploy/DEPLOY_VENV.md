# SparseUNet4D — deployment & demo guide (venv, no conda)

Run the LiDAR moving-object-segmentation (MOS) model live — on **replayed KITTI
data**, in the **Husky Gazebo simulation**, or on a **real Husky robot** —
alongside an existing **ROS 2 Humble** install, **without conda** and **without
breaking** the system Python or ROS. All ML dependencies live in one throwaway
venv; only a single, removable system package (`nvidia-cuda-toolkit`) is added.

Verified on: **Ubuntu 22.04**, **RTX 3050 Ti Laptop (4 GB VRAM, Ampere sm_86)**,
system **Python 3.10.12**, NVIDIA driver present, ROS 2 Humble present.

---

## 1. What lives where (the isolation contract)

| Component | Location | Removal |
|---|---|---|
| torch, MinkowskiEngine, numpy, kiss-icp | `~/mos_venv/` (venv) | `rm -rf ~/mos_venv` |
| `nvcc` + CUDA headers (to *build* MinkowskiEngine) | `nvidia-cuda-toolkit` (system) | `sudo apt remove nvidia-cuda-toolkit` |
| `gcc-10` (host compiler nvcc needs) | system, *additive* next to gcc-11 | `sudo apt remove gcc-10 g++-10` |
| model code | `~/sparseunet4d/` | `rm -rf ~/sparseunet4d` |
| model weights `best.pt` | copied in by hand (not in git) | delete the file |

Nothing overwrites a system library. ROS 2 and Husky keep importing from
`/usr/lib/python3.10`, which the venv never writes to.

---

## 2. One-time setup

Prereqs: `nvidia-smi` prints your GPU; `/opt/ros/humble/setup.bash` exists;
`python3.10 --version` is 3.10 (this is what ROS `rclpy` is built against — the
venv **must** come from it).

### Quick path (recommended) — one command

```bash
git clone -b sota-push https://github.com/subhransu10/sparseunet4d ~/sparseunet4d
cd ~/sparseunet4d
bash deploy/setup_venv.sh          # installs everything into ~/mos_venv (~20 min)
#   then make sure that the checkpoint is present in runs/consistency_ft/best.pt (scp or USB)
```
`setup_venv.sh` is idempotent (safe to re-run), auto-detects your GPU's compute
capability, builds MinkowskiEngine with `MAX_JOBS=1` (no laptop OOM-freeze), and
writes `~/activate_mos.sh` — source that before every run (§3). If it finishes
cleanly you can skip to §4. The manual steps below are the same thing, spelled
out, if you prefer to run them yourself or the script fails partway.

### Manual steps (what the script does)

```bash
# 2.1 system packages (additive, removable, do NOT touch ROS/driver)
sudo apt update
sudo apt install -y python3.10-venv git wget gcc-10 g++-10 libopenblas-dev
sudo apt install -y nvidia-cuda-toolkit            # provides /usr/bin/nvcc
which nvcc && nvcc --version | tail -3             # must print a CUDA version

# 2.2 venv from system python3.10
python3.10 -m venv ~/mos_venv
source ~/mos_venv/bin/activate
pip install --upgrade pip wheel

# 2.3 PyTorch 1.12.1 + cu113 (canonical zero-patch MinkowskiEngine combo)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 2.4 build/runtime deps
pip install "numpy<2" pyyaml scipy ninja        # numpy<2 is required
pip install kiss-icp                             # optional: LiDAR self-odometry

# 2.5 build MinkowskiEngine 0.5.4  — ONE job at a time or the laptop OOM-freezes
export CUDA_HOME=/usr PATH=/usr/bin:$PATH CC=gcc-10 CXX=g++-10
export TORCH_CUDA_ARCH_LIST="8.6"                # 3050 Ti = sm_86; use your GPU's arch
export MAX_JOBS=1                                # CRITICAL on laptops
git clone https://github.com/NVIDIA/MinkowskiEngine.git ~/MinkowskiEngine
cd ~/MinkowskiEngine && python setup.py install --blas=openblas --force_cuda
# success ends: "Finished processing dependencies for MinkowskiEngine==0.5.4"

# 2.6 model code + weights
git clone -b sota-push https://github.com/subhransu10/sparseunet4d ~/sparseunet4d
mkdir -p ~/sparseunet4d/runs/consistency_ft
#   copy best.pt (scp or USB) into ~/sparseunet4d/runs/consistency_ft/best.pt

# 2.7 verify the whole stack imports together
source /opt/ros/humble/setup.bash
python - <<'PY'
import torch, MinkowskiEngine as ME, rclpy
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("MinkowskiEngine:", ME.__version__, "| rclpy OK")
PY
```

**Code note:** the `sota-push` branch must include the deployment fixes:
Gazebo-compatible cloud reader (`read_points`, NaN/inf filtering), degenerate-
quaternion guard, pose–scan time synchronization, and the configurable
`SU4D_THRESHOLD`. If your clone predates them, apply the patches before running.

---

## 3. Launch preamble — run in EVERY node terminal

`setup_venv.sh` generated this helper — just source it (it does all three lines):
```bash
source ~/activate_mos.sh
```
Equivalent to:
```bash
source /opt/ros/humble/setup.bash
source ~/mos_venv/bin/activate
export SU4D_BACKEND=me          # REQUIRED — selects MinkowskiEngine, not the CPU stand-in
```

> `SU4D_BACKEND=me` is mandatory and easy to forget after a reboot/new terminal.
> Without it the model builds the mock backend (`lin.weight`) and the real
> checkpoint (`conv.kernel`) fails to load with every weight "missing".

`SU4D_THRESHOLD` sets the moving-probability cutoff (default = the checkpoint's
tuned value). Higher = fewer false positives (precision); lower = more range/
recall. In-domain 64-beam: `0.5`. Sparse 16-beam robot LiDAR: `0.7–0.8`.

---

## 4. Running the model — three scenarios

### A) KITTI seq-08 replay  (best quality — in-domain, ground-truth poses)

This is the exact benchmark data (dense 64-beam + perfect poses). No KISS-ICP
needed. Put `kitti_replay.py` in `~/` and the sequence folder anywhere (needs
`velodyne/*.bin`, `poses.txt`, `calib.txt`, `times.txt`).

```bash
# Terminal 1 — replay KITTI as /points + /odom + (map->velodyne) TF
source /opt/ros/humble/setup.bash && source ~/mos_venv/bin/activate
python ~/kitti_replay.py --seq-dir /path/to/sequences/08 --rate 10 --loop

# Terminal 2 — MOS node (GT poses; no KISS-ICP)
source /opt/ros/humble/setup.bash && source ~/mos_venv/bin/activate
export SU4D_BACKEND=me
cd ~/sparseunet4d
SU4D_THRESHOLD=0.5 python mos_node.py --ros-args \
  -p config:=$HOME/sparseunet4d/configs/consistency_ft.yaml \
  -p ckpt:=$HOME/sparseunet4d/runs/consistency_ft/best.pt \
  -p propagate:=true \
  -r /sparseunet4d_mos/points:=/points \
  -r /sparseunet4d_mos/odom:=/odom
```
RViz: Fixed Frame **`map`**, Views→Target Frame **`velodyne`** (see §5).

### B) Husky Gazebo simulation

```bash
# Terminal 1 — the sim, PLAIN system ROS (no venv)
source /opt/ros/humble/setup.bash
ros2 launch clearpath_gz simulation.launch.py     # or your usual sim launch

# Terminal 2 — MOS node
source /opt/ros/humble/setup.bash && source ~/mos_venv/bin/activate
export SU4D_BACKEND=me
cd ~/sparseunet4d
SU4D_THRESHOLD=0.7 python mos_node.py --ros-args \
  -p config:=$HOME/sparseunet4d/configs/consistency_ft.yaml \
  -p ckpt:=$HOME/sparseunet4d/runs/consistency_ft/best.pt \
  -p propagate:=true \
  -r /sparseunet4d_mos/points:=/velodyne_points \
  -r /sparseunet4d_mos/odom:=/platform/odom/filtered
```
Confirm topics with `ros2 topic list | grep -Ei "point|odom"`. RViz Fixed Frame
**`base_link`** (parked demo) — the sim has no `odom` TF, so `map`/`odom` won't
work there unless you run an odom→base_link broadcaster.

### C) Real Husky robot

Same as the sim, but with the real sensor topics. Find them on the robot:
```bash
ros2 topic list | grep -Ei "point|scan|velodyne|lidar|odom"
```
Then (typical Clearpath topics shown — substitute yours):
```bash
source /opt/ros/humble/setup.bash && source ~/mos_venv/bin/activate
export SU4D_BACKEND=me
cd ~/sparseunet4d
SU4D_THRESHOLD=0.7 python mos_node.py --ros-args \
  -p config:=$HOME/sparseunet4d/configs/consistency_ft.yaml \
  -p ckpt:=$HOME/sparseunet4d/runs/consistency_ft/best.pt \
  -p propagate:=true \
  -r /sparseunet4d_mos/points:=/sensors/lidar3d_0/points \
  -r /sparseunet4d_mos/odom:=/platform/odom/filtered
```
- **No usable odom?** add `-p use_kiss_icp:=true` to self-localize from the LiDAR.
- **Run onboard or offboard.** To run the node on a laptop while the robot
  publishes, put both on the same `ROS_DOMAIN_ID` and network.
- **Domain gap:** the model is trained on 64-beam KITTI. A 16/32-beam Husky
  LiDAR is sparser → expect lower accuracy and range. The pipeline is correct;
  for a polished result, fine-tune on a little labelled robot data.

Node outputs (all scenarios):
`/sparseunet4d_mos/points_labeled` (all points + `moving`/`moving_prob` fields)
and `/sparseunet4d_mos/points_moving` (moving points only).

---

## 5. RViz visualization

```bash
rviz2 -d ~/mos_demo.rviz      # preset: green mover spheres + dim static context
```
- **Displays:** PointCloud2 on `/sparseunet4d_mos/points_moving` (Style *Spheres*,
  Size ~0.3 m, FlatColor green); a dim PointCloud2 on the raw cloud for context.
- **World-fixed view with a following camera** (KITTI / anything with a world TF):
  Fixed Frame **`map`**, then **Views panel → Target Frame → `velodyne`** so the
  camera rides the vehicle while the world stays put. Set **Decay Time ~0.5** so
  movers leave clean trails and static points overlap.
- **Sensor-centric view** (no world TF — bare rosbags): Fixed Frame = the
  cloud's own `frame_id`, **Decay Time 0** (else everything smears).
- If `map` looks blank, the cloud is at its world coordinate off-screen — set
  Target Frame `velodyne` and lower Distance to ~40; it snaps onto the vehicle.
- Disable the RobotModel display if there's no `/robot_description`.

---

## 6. Errors you may hit (and the fix)

**`Error 804: forward compatibility was attempted on non supported HW` (CUDA init).**
The NVIDIA driver was updated in the background; the running kernel module no
longer matches. **Reboot** (`sudo reboot`). Confirm with `nvidia-smi` — a
"Driver/library version mismatch" message is the tell.

**Checkpoint load fails, every weight "missing" (`stem…`, `encoders…`).**
You forgot `export SU4D_BACKEND=me`. Mock backend (`lin.weight`) can't load ME
weights (`conv.kernel`). Set it before running Python. (Gone after every reboot.)

**Node runs but nothing publishes; `ros2 node info` shows it on `/<node>/points`.**
The `-r ~/points:=…` remap's `~` was eaten by the shell. Use fully-resolved
names: `-r /sparseunet4d_mos/points:=/velodyne_points`. No pose = every scan
dropped, so a wrong odom remap = total silence.

**`All fields need to have the same datatype … Use read_points()`.**
Gazebo/real velodyne clouds have mixed-dtype fields. The fixed `_read_xyzi`
uses `pc2.read_points` + finite filtering. Ensure your `mos_node.py` has it.

**Static walls flagged green while driving (false positives).**
Ego-motion isn't perfectly cancelled — pose error or pose/scan timing mismatch.
Fixes: the pose–scan time-sync patch (interpolate odom to the scan stamp),
drive slower, raise `SU4D_THRESHOLD`. Parked = clean is expected.

**`SO3::exp failed … nan` (KISS-ICP).**
KISS-ICP failed to track (too sparse a cloud). Use a real odom topic instead,
or denser LiDAR. It works far better on structured 64-beam scenes.

**PC freezes during MinkowskiEngine build.** Parallel `nvcc` OOM. Reboot, rebuild
with `MAX_JOBS=1`; add an 8 GB swapfile if RAM < 16 GB.

**`No matching distribution found for nvidia-cudart-cu11`.** That package name
doesn't exist — get `nvcc` from `sudo apt install nvidia-cuda-toolkit`.

**Runtime OOM on 4 GB VRAM.** Full 64-beam over 5 frames can exceed 4 GB (KITTI
replay peaks ~0.5 GB in practice, so usually fine). If it OOMs: fewer `n_frames`,
larger `voxel_size`, or run the node on a bigger GPU.

**Harmless noise:** pip's `generate-parameter-library-py requires jinja2` (system
ROS pkgs), `NO_PUBKEY` apt warnings (third-party repos), a CUDA version-mismatch
*warning* during the ME build (11.5 nvcc vs cu113 torch — ABI-compatible).

**GitHub "Unverified" commits.** Cosmetic (missing signature); set up SSH commit
signing if you want the badge.

---

## 7. Version notes
- `nvcc` from `nvidia-cuda-toolkit` on 22.04 is ~CUDA 11.5; torch is cu113 — the
  11.x ABI is compatible, the build warning is harmless.
- `TORCH_CUDA_ARCH_LIST` = your GPU's capability (3050 Ti = `8.6`;
  `python -c "import torch; print(torch.cuda.get_device_capability())"`).
- Warm-up: the widest temporal offset is 8, so the first ~8 scans give partial
  predictions while the buffer fills — by design.
