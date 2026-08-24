# SparseUNet4D on Husky: simple deployment guide

This guide runs **standalone SparseUNet4D** with an existing Husky ROS 2
simulation or a real Husky. It is tailored to an RTX 3050 Ti with 4 GB VRAM
and assumes MinkowskiEngine already works on that computer.

The node consumes:

- LiDAR: `sensor_msgs/msg/PointCloud2`
- Pose: `nav_msgs/msg/Odometry`

It publishes:

- `/sparseunet4d_mos/points_labeled` — all points with `moving` and
  `moving_prob` fields
- `/sparseunet4d_mos/points_moving` — moving points only

The model needs five scans with offsets `[0, 1, 2, 4, 8]`, so output starts
after its history is filled. This guide disables the MapMOS/hybrid path with
`propagate:=false`.

> Safety: on a real robot, use this as a perception-only experiment first.
> Do not connect predictions directly to steering or emergency-stop logic.

## 1. Clone the correct repository and branch

On the Husky computer:

```bash
cd ~
git clone --branch icra27 --single-branch \
  https://github.com/subhransu10/sparseunet4d.git
cd ~/sparseunet4d
```

If the repository was copied from your hard disk instead:

```bash
cd ~/sparseunet4d
git branch --show-current
```

The result must be `icra27`. The pending result-only Git commit is not needed
to run the robot.

## 2. Copy the trained checkpoint

Git does not contain the large checkpoint. Copy this directory from your
backup into the cloned repository:

```text
runs/icra27_standard08_aggregate_ft/
```

At minimum these two files must exist:

```text
runs/icra27_standard08_aggregate_ft/best.pt
runs/icra27_standard08_aggregate_ft/config.yaml
```

Verify it:

```bash
cd ~/sparseunet4d
test -f runs/icra27_standard08_aggregate_ft/best.pt \
  && test -f runs/icra27_standard08_aggregate_ft/config.yaml \
  && echo "checkpoint and config found" \
  || echo "ERROR: checkpoint or config missing"
```

## 3. Create the RTX 3050 Ti deployment configuration

The original 51.2 m point range may exceed 4 GB VRAM. Make a deployment copy
and initially limit it to 25.6 m:

```bash
cd ~/sparseunet4d
cp runs/icra27_standard08_aggregate_ft/config.yaml \
   configs/deploy_husky_3050ti.yaml

sed -i 's/point_range: 51.2/point_range: 25.6/' \
  configs/deploy_husky_3050ti.yaml

grep -E 'point_range|n_frames|frame_offsets|feat_rep' \
  configs/deploy_husky_3050ti.yaml
```

Do not change `n_frames`, `frame_offsets`, or the feature representation. If
CUDA still runs out of memory, change only `point_range` to `20.0`.

## 4. Activate the working environment

Use the environment in which MinkowskiEngine previously worked. For example:

```bash
conda activate torch5090
source /opt/ros/humble/setup.bash
cd ~/sparseunet4d

export SU4D_BACKEND=me
export PYTHONPATH="$HOME/MinkowskiEngine:$HOME/sparseunet4d:${PYTHONPATH:-}"
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

If MinkowskiEngine was installed as a normal Python package, its source path
may be omitted:

```bash
export PYTHONPATH="$HOME/sparseunet4d:${PYTHONPATH:-}"
```

Verify everything before starting Gazebo:

```bash
python - <<'PY'
import torch
import MinkowskiEngine
import rclpy
from sparseunet4d.models.backend import backend

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Sparse backend:", backend())
assert torch.cuda.is_available(), "CUDA is unavailable"
assert backend() == "me", "MinkowskiEngine backend is not active"
print("Environment verification passed")
PY

nvidia-smi
```

### First-time PC without MinkowskiEngine

Skip this subsection on your RTX 3050 Ti because MinkowskiEngine already
works there. On a different computer, first install the NVIDIA driver, a CUDA
enabled PyTorch build, compiler tools and OpenBLAS. Then build MinkowskiEngine
against the **same CUDA version used by PyTorch**. Follow the maintained
[official MinkowskiEngine installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation)
rather than copying version-specific commands from another computer.

Do not continue until this passes inside the same environment used for ROS:

```bash
python - <<'PY'
import torch
import MinkowskiEngine
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA usable:", torch.cuda.is_available())
print("MinkowskiEngine import passed")
assert torch.cuda.is_available()
PY
```

## 5. Start your existing Husky simulation

Open terminal 1, source your Husky workspace, and launch the simulator using
the command that already works for your package:

```bash
source /opt/ros/humble/setup.bash
source ~/husky_ws/install/setup.bash
ros2 launch <your_husky_package> <your_existing_gazebo_launch_file>.launch.py
```

Do not copy the placeholder names literally. Use your existing package and
launch file.

## 6. Find the LiDAR and odometry topics

In terminal 2:

```bash
source /opt/ros/humble/setup.bash
source ~/husky_ws/install/setup.bash

ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
```

Check the candidate topics:

```bash
ros2 topic type /velodyne_points
ros2 topic type /odom
ros2 topic hz /velodyne_points
ros2 topic hz /odom
```

The types must be:

```text
sensor_msgs/msg/PointCloud2
nav_msgs/msg/Odometry
```

Replace the example names below with the topics found on your robot:

```bash
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/odom
```

Important: odometry must describe the LiDAR pose, or the base pose with the
correct rigid LiDAR extrinsic already applied. A wrong pose or timestamp makes
static walls look moving, especially during turns.

## 7. Run SparseUNet4D in Gazebo

In terminal 3:

```bash
conda activate torch5090
source /opt/ros/humble/setup.bash
source ~/husky_ws/install/setup.bash
cd ~/sparseunet4d

export SU4D_BACKEND=me
export PYTHONPATH="$HOME/MinkowskiEngine:$HOME/sparseunet4d:${PYTHONPATH:-}"
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MODEL_CONFIG="$HOME/sparseunet4d/configs/deploy_husky_3050ti.yaml"
export MODEL_CKPT="$HOME/sparseunet4d/runs/icra27_standard08_aggregate_ft/best.pt"
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/odom
export SU4D_THRESHOLD=0.92

python mos_node.py --ros-args \
  -p config:="$MODEL_CONFIG" \
  -p ckpt:="$MODEL_CKPT" \
  -p device:=cuda \
  -p propagate:=false \
  -p pipeline:=true \
  -p use_sim_time:=true \
  -r /sparseunet4d_mos/points:="$CLOUD_TOPIC" \
  -r /sparseunet4d_mos/odom:="$ODOM_TOPIC"
```

`pipeline:=true` drops stale queued scans instead of exhausting the 4 GB GPU.
The threshold `0.92` is the validated standalone aggregate-model threshold.

## 8. Confirm and visualize the output

In terminal 4:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /sparseunet4d_mos/points_moving
ros2 topic info /sparseunet4d_mos/points_labeled -v
ros2 topic info /sparseunet4d_mos/points_moving -v
```

Open RViz:

```bash
rviz2 --ros-args -p use_sim_time:=true
```

Add two `PointCloud2` displays:

1. input cloud, such as `/velodyne_points`;
2. `/sparseunet4d_mos/points_moving`.

Set RViz's fixed frame to the frame used by your Husky simulation, commonly
`odom` or `map`.

## 9. Run on the real Husky

Use the same command with three changes:

1. launch the real LiDAR and localization drivers;
2. select their real cloud and odometry topics;
3. set `use_sim_time:=false`.

```bash
python mos_node.py --ros-args \
  -p config:="$MODEL_CONFIG" \
  -p ckpt:="$MODEL_CKPT" \
  -p device:=cuda \
  -p propagate:=false \
  -p pipeline:=true \
  -p use_sim_time:=false \
  -r /sparseunet4d_mos/points:="$CLOUD_TOPIC" \
  -r /sparseunet4d_mos/odom:="$ODOM_TOPIC"
```

First test while the Husky is stationary, then drive slowly in a controlled
area with an operator at the emergency stop.

### If reliable odometry is unavailable

The node can estimate motion with KISS-ICP:

```bash
python -m pip install kiss-icp

python mos_node.py --ros-args \
  -p config:="$MODEL_CONFIG" \
  -p ckpt:="$MODEL_CKPT" \
  -p device:=cuda \
  -p propagate:=false \
  -p pipeline:=true \
  -p use_sim_time:=false \
  -p use_kiss_icp:=true \
  -r /sparseunet4d_mos/points:="$CLOUD_TOPIC"
```

## 10. Fast diagnosis

### `No module named MinkowskiEngine`

You activated the wrong Python environment or omitted its source path:

```bash
conda activate torch5090
export PYTHONPATH="$HOME/MinkowskiEngine:$HOME/sparseunet4d:${PYTHONPATH:-}"
python -c "import MinkowskiEngine; print('MinkowskiEngine OK')"
```

For a new PC without MinkowskiEngine, install a CUDA/PyTorch-compatible build
using MinkowskiEngine's official installation instructions. Confirm the import
above before attempting to run SparseUNet4D. A CPU/mock backend is not a valid
deployment substitute.

### Backend says `mock` instead of `me`

```bash
export SU4D_BACKEND=me
python -c "from sparseunet4d.models.backend import backend; print(backend())"
```

### CUDA out of memory

1. Close GPU-heavy applications and RViz.
2. Keep `pipeline:=true`.
3. Change `point_range` from `25.6` to `20.0`.
4. Start the model first, then RViz.
5. Watch memory with `watch -n 1 nvidia-smi`.

Do not change frame count, offsets, or feature width just to suppress OOM.

### No output topic or no messages

```bash
ros2 node list
ros2 node info /sparseunet4d_mos
ros2 topic hz "$CLOUD_TOPIC"
ros2 topic hz "$ODOM_TOPIC"
ros2 topic echo "$ODOM_TOPIC" --once
```

Check the full remap names exactly as shown in the launch command. Do not use
shell paths such as `~/points` as ROS topic names.

### Static objects are marked moving

This is usually pose alignment, timestamp synchronization, or the LiDAR-to-base
extrinsic—not the motion threshold. Check those three items before changing
`SU4D_THRESHOLD`.

### Inference is too slow

Keep `pipeline:=true`, reduce `point_range`, close RViz while measuring, and
confirm the process is using the NVIDIA GPU with `nvidia-smi`.

### Checkpoint loading fails

Confirm all four items:

```bash
git branch --show-current
test -f "$MODEL_CONFIG" && echo config-ok
test -f "$MODEL_CKPT" && echo checkpoint-ok
python -c "from sparseunet4d.models.backend import backend; print(backend())"
```

Expected: branch `icra27`, both files present, backend `me`.

## 11. Save one reproducibility record

After a successful run:

```bash
mkdir -p ~/sparseunet4d/runs/husky_deployment_record

git rev-parse HEAD \
  > ~/sparseunet4d/runs/husky_deployment_record/git_commit.txt

sha256sum "$MODEL_CONFIG" "$MODEL_CKPT" \
  > ~/sparseunet4d/runs/husky_deployment_record/input_sha256.txt

nvidia-smi \
  > ~/sparseunet4d/runs/husky_deployment_record/nvidia_smi.txt

ros2 topic list -t \
  > ~/sparseunet4d/runs/husky_deployment_record/ros_topics.txt
```

Also record the Husky package/commit, simulation world, LiDAR model and rate,
cloud topic, pose source, point range, threshold, and whether the run used
Gazebo time or real time.

## Minimal checklist

- [ ] Clone `https://github.com/subhransu10/sparseunet4d.git`, branch `icra27`.
- [ ] Copy `runs/icra27_standard08_aggregate_ft/best.pt` from backup.
- [ ] Create `configs/deploy_husky_3050ti.yaml` with `point_range: 25.6`.
- [ ] Activate the existing MinkowskiEngine environment and verify backend `me`.
- [ ] Start Husky simulation and identify cloud/odometry topics.
- [ ] Run `mos_node.py` with `propagate:=false` and `pipeline:=true`.
- [ ] Confirm `/sparseunet4d_mos/points_moving` in RViz.
- [ ] Only then repeat with real sensors and `use_sim_time:=false`.
