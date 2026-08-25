# SparseUNet4D on Husky and Gazebo

The published image is self-contained: it includes ROS 2 Humble, PyTorch
`1.12.1+cu113`, MinkowskiEngine `0.5.4`, the SparseUNet4D source and configs,
and the released `best.pt`. No Python environment or model mount is required.

The x86-64 host needs Docker, an NVIDIA driver, and NVIDIA Container Toolkit.
Confirm that `nvidia-smi` works before continuing. MinkowskiEngine is compiled
for CUDA compute capability 8.6 (including RTX 3050 Ti and RTX 3090), with PTX
included for compatible newer GPUs.

## Pull

```bash
docker pull ghcr.io/subhransu10/sparseunet4d:latest
```

Optional version check:

```bash
docker run --rm --gpus all \
  ghcr.io/subhransu10/sparseunet4d:latest \
  python3 -c 'import torch, MinkowskiEngine as ME; print(torch.__version__, torch.version.cuda, ME.__version__); print(torch.cuda.get_device_name())'
```

Expected versions are `1.12.1+cu113`, CUDA `11.3`, and MinkowskiEngine `0.5.4`.

## Run with Gazebo

Start Husky Gazebo on the host. Find its point-cloud and odometry topics:

```bash
ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
ros2 topic type /velodyne_points
ros2 topic type /odom
```

The inputs must be `sensor_msgs/msg/PointCloud2` and
`nav_msgs/msg/Odometry`. Then run the image, replacing the two example topic
names if necessary:

```bash
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/odom
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

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

The container uses host networking to join the host's ROS 2 DDS domain. The
five-frame temporal window warms up during the first eight scans.

## Run on a real Husky

Use the same command with the robot's LiDAR and lidar-localization odometry
topics, and change `use_sim_time` to `false`. Start stationary and test at low
speed in a controlled area. Odometry timestamps and the LiDAR pose/extrinsic
must be correct; wheel odometry alone may not be accurate enough.

This is experimental perception output. Do not connect it directly to
steering, braking, or emergency-stop control.

## Check output

On the host:

```bash
ros2 node list
ros2 topic hz /sparseunet4d_mos/points_moving
ros2 topic info /sparseunet4d_mos/points_labeled
```

The node publishes:

- `/sparseunet4d_mos/points_labeled`: the input cloud with `moving` and
  `moving_prob` fields;
- `/sparseunet4d_mos/points_moving`: moving points only.

If ROS topics are not visible, confirm `--network host` and matching
`ROS_DOMAIN_ID` values. If the GPU is not visible, fix NVIDIA Container
Toolkit on the host before debugging the model.
