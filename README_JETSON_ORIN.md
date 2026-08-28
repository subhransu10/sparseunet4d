# Jetson AGX Orin Docker deployment

This is the tested real-robot path for Jetson AGX Orin Developer Kit,
JetPack 6.2 / L4T R36.5, and ROS domain `30`.

The regular GHCR `latest` image is `amd64` and cannot run on Jetson. Build the
Jetson image locally from this repository. The resulting container includes
ROS 2 Humble, CUDA 12.6, PyTorch 2.4, MinkowskiEngine 0.5.4, source, configs,
and the released `best.pt`.

Do not follow `INSTALLATION.md`, create a Python virtual environment, compile
MinkowskiEngine on the host, or download the checkpoint separately. Cloning is
needed only to provide the Docker build context.

## 1. Check the Jetson

```bash
uname -m
cat /proc/device-tree/model
cat /etc/nv_tegra_release
docker --version
```

Expected architecture is `aarch64`; the tested device is Jetson AGX Orin with
JetPack 6.2 / L4T R36.5 and NVIDIA Container Runtime.

## 2. Clone and build once

```bash
mkdir -p ~/ros_workspaces/ros2/suba_ws/src
cd ~/ros_workspaces/ros2/suba_ws/src
git clone https://github.com/subhransu10/sparseunet4d.git
cd sparseunet4d

docker build --progress=plain \
  --file Dockerfile.jetson \
  --tag ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 .
```

The first build downloads the L4T base image and compiles MinkowskiEngine, so
it can take a long time. Docker build does not require GPU access. Subsequent
source-only rebuilds normally reuse the expensive cached layers.

This command creates a local image with a GHCR-style tag; it does not pull or
push the Jetson image.

## 3. Verify the container GPU

```bash
docker run --rm --runtime nvidia \
  ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 \
  python3 -c 'import torch, MinkowskiEngine as ME; print(torch.__version__, torch.version.cuda, ME.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

Expected output includes PyTorch `2.4`, CUDA `12.6`, MinkowskiEngine `0.5.4`,
`True`, and `Orin`.

## 4. Enable the tested performance mode

The initial robot test used the 30 W profile: 8 CPUs, 4 GPU TPCs and a 612 MHz
GPU, producing approximately `1.2-1.5 Hz`. The following host settings enabled
12 CPUs at 2.2 GHz, 8 GPU TPCs at 1.3 GHz and maximum memory clocks, producing
approximately `4.2 Hz` end to end on the measured robot cloud:

```bash
sudo nvpmodel -m 0
```

If prompted, reboot. After the reboot, run:

```bash
sudo jetson_clocks
sudo nvpmodel -q
sudo jetson_clocks --show
```

Expected power mode is `MAXN`. Use the proper Jetson power supply and cooling.
The major speed improvement came from `MAXN` and `jetson_clocks`; the sensor
adaptations below improve detection correctness, not throughput. Actual rate
depends on the scene's active voxel count. The power profile persists, but run
or verify `jetson_clocks` again after a reboot before benchmarking.

## 5. Confirm robot topics

```bash
export ROS_DOMAIN_ID=30

ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
ros2 topic type /velodyne_points
ros2 topic type /platform/odom
```

The tested inputs are:

```text
/velodyne_points   sensor_msgs/msg/PointCloud2
/platform/odom     nav_msgs/msg/Odometry
```

If the robot uses different topics, change only the two remap values at the
end of the run command.

## 6. Run on the robot

After every reboot, set the ROS domain again in the new shell:

```bash
cd ~/ros_workspaces/ros2/suba_ws/src/sparseunet4d
export ROS_DOMAIN_ID=30

docker stop sparseunet4d-mos 2>/dev/null || true

docker run --rm --name sparseunet4d-mos \
  --runtime nvidia --network host --ipc host \
  -e ROS_DOMAIN_ID=30 \
  -e SU4D_THRESHOLD=0.3 \
  -e OMP_NUM_THREADS=6 \
  ghcr.io/subhransu10/sparseunet4d:jetson-orin-jp62 \
  python3 /opt/sparseunet4d/mos_node.py --ros-args \
    -p config:=/opt/sparseunet4d/configs/deploy_jetson_orin.yaml \
    -p ckpt:=/opt/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt \
    -p device:=cuda -p propagate:=false -p pipeline:=true \
    -p intensity_scale:=255.0 \
    -p projection_height:=16 -p projection_width:=2048 \
    -p fov_down_deg:=-15.0 -p fov_up_deg:=15.0 \
    -p use_sim_time:=false \
    -r /sparseunet4d_mos/points:=/velodyne_points \
    -r /sparseunet4d_mos/odom:=/platform/odom
```

Why the robot-specific parameters are required:

- `intensity_scale:=255.0` converts the measured `0-123` intensity range to
  the normalized range expected by the checkpoint.
- The measured 16-beam LiDAR spans approximately `-15` to `+15` degrees,
  unlike the SemanticKITTI HDL-64E projection.
- The node records every incoming 10 Hz scan in temporal history even though
  inference publishes more slowly, preserving trained frame offsets
  `[1,2,4,8]`.
- The deployment config keeps the trained 0.1 m voxels and five-frame model,
  with a practical 25.6 m range. A tested 15 m crop did not improve the indoor
  runtime, so it is not recommended.

## 7. Verify output

In another ROS-sourced terminal on the robot:

```bash
export ROS_DOMAIN_ID=30

ros2 topic hz /sparseunet4d_mos/points_labeled
ros2 topic hz /sparseunet4d_mos/points_moving
docker logs --tail 30 sparseunet4d-mos
```

The logs report input normalization, latency, maximum motion probability and
moving-point counts. `/points_moving` is published only when moving points are
present, so its rate may be lower than `/points_labeled`.

For exact label alignment in RViz, display the full labeled cloud and the
moving-only cloud from the same inference timestamp:

```bash
rviz2 -d sparseunet4d_moving_red.rviz
```

When inference runs on a separate GPU PC over Ethernet, use the preset matching
the tested Husky display (Reliable/Volatile QoS, `lidar3d_0_link`, and a white
moving-points overlay):

```bash
export ROS_DOMAIN_ID=30
rviz2 -d deploy/husky_remote_pc.rviz
```

Set RViz Fixed Frame to `odom` or the fixed frame used by the robot's TF tree.
Do not overlay delayed labels on the live 10 Hz cloud when evaluating spatial
accuracy: the approximately 240 ms inference latency makes those timestamps
different.

## Troubleshooting

- `Couldn't parse remap rule ...:=` means a topic environment variable was
  empty. The command above uses literal topic names to avoid this problem.
- `container name ... is already in use` means an older container exists; run
  `docker stop sparseunet4d-mos` before starting it again.
- `exec: -v: invalid option` means a Docker option was placed after the image
  name. All `-v`, `-e`, networking and runtime options must appear before the
  image name.
- No ROS topics: verify `ROS_DOMAIN_ID=30` on both host and container and keep
  `--network host`.
- No GPU: repair NVIDIA Container Runtime before debugging the model.
