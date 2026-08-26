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
  ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 \
  python3 /opt/sparseunet4d/mos_node.py --ros-args \
    -p config:=/opt/sparseunet4d/configs/pretrained_semantickitti.yaml \
    -p ckpt:=/opt/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt \
    -p device:=cuda -p propagate:=false -p pipeline:=true \
    -p use_sim_time:=false \
    -r /sparseunet4d_mos/points:="$CLOUD_TOPIC" \
    -r /sparseunet4d_mos/odom:="$ODOM_TOPIC"
```

