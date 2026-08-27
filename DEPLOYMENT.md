# Source/venv running and deployment

This guide covers offline verification, SemanticKITTI replay, and the ROS 2
streaming node. Complete [INSTALLATION.md](INSTALLATION.md) and download the
checkpoint first.

This page assumes a host-native source installation. Docker users should use
the separate [x86/Gazebo](README_HUSKY_GAZEBO.md) or
[Jetson](README_JETSON_ORIN.md) guide and should not repeat these installation
steps inside or outside the container.

## Common environment

Open a new terminal and run:

```bash
source ~/activate_mos.sh
cd ~/sparseunet4d

export MODEL_CONFIG="$PWD/configs/pretrained_semantickitti.yaml"
export MODEL_CKPT="$PWD/checkpoints/sparseunet4d_semantickitti/best.pt"
```

`SU4D_BACKEND=me` is set by `activate_mos.sh`. The model needs time-synchronized
LiDAR scans and sensor poses in one fixed coordinate frame. Poor pose estimates
make static surfaces appear to move.

## SemanticKITTI replay check

A sequence directory must contain `velodyne/`, `labels/`, `calib.txt`, and
`poses.txt`. Edit `dataset.root` and `dataset.semantic_yaml` in a local copy of
`configs/pretrained_semantickitti.yaml`, then run:

```bash
python mos_inference.py \
  --replay-test \
  --config "$MODEL_CONFIG" \
  --ckpt "$MODEL_CKPT" \
  --device cuda \
  --frames 100
```

## ROS 2 node

The node subscribes to a `sensor_msgs/msg/PointCloud2` scan and a
`nav_msgs/msg/Odometry` pose. It publishes:

- `/sparseunet4d_mos/points_labeled`: all points plus `moving` and
  `moving_prob` fields;
- `/sparseunet4d_mos/points_moving`: moving points only.

First discover the actual robot or simulator topics:

```bash
ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
ros2 topic type /velodyne_points
ros2 topic type /platform/odom
```

Then launch the node, replacing the example topic names when necessary:

```bash
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/platform/odom
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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

For a real robot, omit `-p use_sim_time:=true`. The `pipeline` mode keeps the
newest scan and drops stale queued scans instead of allowing latency to grow.
The first eight scans contain partial temporal windows while the buffer warms up.

## RViz

```bash
rviz2 -d sparseunet4d_moving_red.rviz
```

The preset overlays the full labeled cloud in grey and the moving-only topic in
red. Set RViz's Fixed Frame to the fixed frame used by odometry if the preset's
default does not match your system.

## Gazebo and Husky

1. Start the existing Husky/Gazebo stack on the host.
2. Confirm scan and odometry topics are publishing.
3. Run the ROS command above with simulation time enabled.
4. Open the supplied RViz preset.

For an isolated Docker deployment that does not install ML packages on the
robot host, follow [README_HUSKY_GAZEBO.md](README_HUSKY_GAZEBO.md).

## Runtime benchmark

```bash
python deploy/benchmark_runtime.py \
  --config "$MODEL_CONFIG" \
  --ckpt "$MODEL_CKPT" \
  --seq-dir /path/to/SemanticKITTI/dataset/sequences/08 \
  --device cuda \
  --warmup 20 \
  --n 300
```

For separate preprocessing, network, and postprocessing timings:

```bash
python deploy/benchmark_breakdown.py \
  --config "$MODEL_CONFIG" \
  --ckpt "$MODEL_CKPT" \
  --seq-dir /path/to/SemanticKITTI/dataset/sequences/08 \
  --device cuda \
  --warmup 20 \
  --n 100
```

Report end-to-end latency when comparing complete systems; label network-only
latency explicitly when comparing model forward passes.

## Resource-limited computers

On a 4 GB laptop GPU, use `configs/deploy_husky_3050ti.yaml`, close browsers and
other GPU applications, and keep `propagate:=false`. If the whole desktop
freezes, check the previous boot with:

```bash
journalctl -k -b -1 | grep -Ei 'oom|killed process|NVRM|Xid|hang'
```
