# SparseUNet4D on Jetson AGX Orin

This image is for Jetson Orin with JetPack 6.2 / L4T R36.5. The regular
`latest` image is for `x86_64` PCs and will not run on Jetson.

The Jetson image contains ROS 2 Humble, CUDA 12.6, PyTorch 2.4,
MinkowskiEngine 0.5.4, the source, configuration, and released `best.pt`.

## Build once on the Jetson

Clone or update this repository on the Jetson, then run from its root:

```bash
docker build --progress=plain --file Dockerfile.jetson \
  --tag ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 .
```

The first build downloads a large base image and compiles MinkowskiEngine. It
can take a long time. The build does not need GPU access.

## Test the GPU

```bash
docker run --rm --runtime nvidia \
  ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 \
  python3 -c 'import torch, MinkowskiEngine as ME; print(torch.__version__, torch.version.cuda, ME.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

Expected output includes PyTorch `2.4`, CUDA `12.6`, MinkowskiEngine `0.5.4`,
`True`, and `Orin`.

## Run on the robot

Find the robot's point-cloud and odometry topics:

```bash
ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
```

Set those two topic names and run:

```bash
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/platform/odom
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

docker run --rm --name sparseunet4d-mos \
  --runtime nvidia --network host --ipc host \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e SU4D_THRESHOLD=0.3 \
  ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 \
  python3 /opt/sparseunet4d/mos_node.py --ros-args \
    -p config:=/opt/sparseunet4d/configs/pretrained_semantickitti.yaml \
    -p ckpt:=/opt/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt \
    -p device:=cuda -p propagate:=false -p pipeline:=true \
    -p intensity_scale:=255.0 \
    -p projection_height:=16 -p projection_width:=2048 \
    -p fov_down_deg:=-15.0 -p fov_up_deg:=15.0 \
    -p use_sim_time:=false \
    -r /sparseunet4d_mos/points:="$CLOUD_TOPIC" \
    -r /sparseunet4d_mos/odom:="$ODOM_TOPIC"
```

These sensor values match the measured robot cloud: intensity `0-123` and a
roughly `-15` to `+15` degree 16-beam vertical field of view. The node keeps
all incoming 10 Hz scans in its temporal history even when Jetson inference
publishes more slowly.

Check the result from another ROS-sourced terminal:

```bash
ros2 topic hz /sparseunet4d_mos/points_labeled
ros2 topic hz /sparseunet4d_mos/points_moving
docker logs --tail 30 sparseunet4d-mos
```

The logs report the normalized intensity, maximum moving probability, and
maximum number of moving points every five seconds. The `points_moving` topic
is published only when at least one moving point is present.
