# x86_64 Docker: Husky and Gazebo

Use this guide for an NVIDIA `x86_64` PC. The published image is self-contained:
do not clone the repository, follow `INSTALLATION.md`, create a virtual
environment, compile MinkowskiEngine, or download `best.pt` separately.

Jetson Orin is `aarch64` and cannot run this image. Use the separate
[Jetson AGX Orin guide](README_JETSON_ORIN.md).

## 1. Check the PC

```bash
uname -m
nvidia-smi
docker --version
```

Continue when `uname -m` reports `x86_64`, the GPU appears in `nvidia-smi`,
Docker is installed, and NVIDIA Container Toolkit is configured.

## 2. Pull and verify

```bash
docker pull ghcr.io/subhransu10/sparseunet4d:latest

docker run --rm --gpus all \
  ghcr.io/subhransu10/sparseunet4d:latest \
  python3 -c 'import torch, MinkowskiEngine as ME; print(torch.__version__, torch.version.cuda, ME.__version__); print(torch.cuda.get_device_name())'
```

Expected versions are PyTorch `1.12.1+cu113`, CUDA `11.3`, and
MinkowskiEngine `0.5.4`.

## 3. Run with Gazebo

Start Gazebo on the host and identify the point-cloud and odometry topics:

```bash
ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
```

The topics must use `sensor_msgs/msg/PointCloud2` and `nav_msgs/msg/Odometry`.
Set the three values below to match the simulator, then run the container:

```bash
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/odom
export ROS_DOMAIN_ID=0

docker stop sparseunet4d-mos 2>/dev/null || true

docker run --rm --name sparseunet4d-mos \
  --gpus all --network host --ipc host \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  ghcr.io/subhransu10/sparseunet4d:latest \
  python3 /opt/sparseunet4d/mos_node.py --ros-args \
    -p config:=/opt/sparseunet4d/configs/pretrained_semantickitti.yaml \
    -p ckpt:=/opt/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt \
    -p device:=cuda -p propagate:=false -p pipeline:=true \
    -p use_sim_time:=true \
    -r /sparseunet4d_mos/points:="$CLOUD_TOPIC" \
    -r /sparseunet4d_mos/odom:="$ODOM_TOPIC"
```

Host networking lets the container join the host ROS 2 DDS domain. The first
eight scans warm up the temporal window.

## 4. Check output

In another ROS-sourced host terminal:

```bash
ros2 topic hz /sparseunet4d_mos/points_labeled
ros2 topic hz /sparseunet4d_mos/points_moving
```

The labeled topic contains the input points plus `moving` and `moving_prob`.
The moving-only topic is published only when at least one moving point exists.

If topics are invisible, confirm `--network host` and matching
`ROS_DOMAIN_ID` values. If the GPU is invisible, fix NVIDIA Container Toolkit
before debugging SparseUNet4D.
