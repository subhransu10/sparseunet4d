# SparseUNet4D on Husky and Gazebo (Docker deployment)

This guide runs the validated SparseUNet4D moving-object-segmentation node in
ROS 2 Humble, either with Husky Gazebo or on a real Husky. The ML environment
stays inside Docker: do not install PyTorch, CUDA Toolkit, MinkowskiEngine,
NumPy, or SciPy into the robot's system Python.

The node subscribes to:

- `sensor_msgs/msg/PointCloud2` LiDAR scans;
- `nav_msgs/msg/Odometry` poses.

It publishes:

- `/sparseunet4d_mos/points_labeled`: all input points with `moving` and
  `moving_prob` fields;
- `/sparseunet4d_mos/points_moving`: moving points only.

> Safety: treat this as an experimental perception output. Do not connect it
> directly to steering, braking, or emergency-stop control.

## 1. What the robot host needs

The robot PC only needs:

1. Ubuntu 22.04 and ROS 2 Humble for the existing Husky system;
2. a working NVIDIA driver (`nvidia-smi`);
3. Docker Engine;
4. NVIDIA Container Toolkit.

It does **not** need a host installation of PyTorch, MinkowskiEngine, CUDA
Toolkit, a Python virtual environment, or Conda.

Verify Docker GPU access before copying the model:

```bash
nvidia-smi
docker --version
docker info | grep -i runtimes

docker run --rm --gpus all \
  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
  nvidia-smi
```

Stop here if the last command cannot see the GPU. Installing Docker or NVIDIA
Container Toolkit is a host change and must be performed by the robot
administrator if either is absent.

### GPU compatibility

The supplied `mos_venv` contains MinkowskiEngine compiled for Ampere
compute capability `sm_86`. It is directly compatible with the tested RTX
3050 Laptop GPU and RTX 3090. A Jetson or a GPU with another compute capability
needs a matching MinkowskiEngine build; copying this environment is not enough.

## 2. Required deployment artifacts

The cloud benchmark did **not** create a Docker image. It produced a portable
Python environment and benchmark payload. For Husky deployment, prepare these
three artifacts on the development PC:

- the public `master` branch of this repository;
- `mos_venv_sm86.tar`;
- a generic `model/` directory containing `best.pt` and `config.yaml`.

Prepare the generic deployment directory from the original training output:

```bash
mkdir -p /media/suba/Expansion/sparseunet4d_husky/model
cd ~/sparseunet4d
bash deploy/download_model.sh
cp checkpoints/sparseunet4d_semantickitti/best.pt \
  /media/suba/Expansion/sparseunet4d_husky/model/best.pt
cp configs/pretrained_semantickitti.yaml \
  /media/suba/Expansion/sparseunet4d_husky/model/config.yaml
```

Do not copy SemanticKITTI sequence 08 to the robot; it was only benchmark data.

Verify the source and checkpoint on the development PC:

```bash
cd ~/sparseunet4d
git branch --show-current
git rev-parse HEAD

test -f /media/suba/Expansion/sparseunet4d_husky/model/best.pt
test -f /media/suba/Expansion/sparseunet4d_husky/model/config.yaml

sha256sum \
  /media/suba/Expansion/sparseunet4d_husky/model/best.pt \
  /media/suba/Expansion/sparseunet4d_husky/model/config.yaml
```

Save the displayed commit and hashes with the deployment bundle.

## 3. Build the reusable ROS/CUDA runtime image

Build this image once on an Ubuntu x86-64 development computer with Docker.
The image supplies CUDA 11 runtime libraries and ROS 2 Humble. The large ML
environment and model are mounted separately, so rebuilding the image does not
recompile MinkowskiEngine.

Create an empty build directory and save the following as `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 locales python3.10 libopenblas0 libgomp1 \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-humble-ros-base ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
WORKDIR /root/sparseunet4d
```

Build without `--progress=plain` because older Docker legacy builders do not
support that option:

```bash
cd ~/sparseunet4d_husky_image
docker build -t sparseunet4d-husky:runtime .
```

Verify that the image exists:

```bash
docker image inspect sparseunet4d-husky:runtime >/dev/null \
  && echo "runtime image OK"
```

Export it to the external hard disk:

```bash
docker save sparseunet4d-husky:runtime \
  -o /media/suba/Expansion/sparseunet4d-husky_runtime.tar

cd /media/suba/Expansion
sha256sum sparseunet4d-husky_runtime.tar \
  > sparseunet4d-husky_runtime.tar.sha256
```

## 4. Copy and load on the Husky PC

Copy these items to a deployment directory on the robot, using an external
disk or `scp`:

```text
sparseunet4d-husky_runtime.tar
sparseunet4d-husky_runtime.tar.sha256
mos_venv_sm86.tar
sparseunet4d/                       # source repository
model/                              # best.pt and config.yaml
```

Example robot layout:

```text
/opt/sparseunet4d_deploy/
├── mos_venv/
├── sparseunet4d/
├── model/
└── sparseunet4d-husky_runtime.tar
```

Create the layout, extract the environment, and load the image:

```bash
sudo mkdir -p /opt/sparseunet4d_deploy
sudo chown "$USER":"$USER" /opt/sparseunet4d_deploy

cd /opt/sparseunet4d_deploy
tar -xf /path/to/mos_venv_sm86.tar
cp -a /path/to/sparseunet4d ./
cp -a /path/to/model ./

(cd /path/to && sha256sum -c sparseunet4d-husky_runtime.tar.sha256)
docker load -i /path/to/sparseunet4d-husky_runtime.tar
```

Confirm the expected directories:

```bash
test -x /opt/sparseunet4d_deploy/mos_venv/bin/python
test -f /opt/sparseunet4d_deploy/sparseunet4d/mos_node.py
test -f /opt/sparseunet4d_deploy/model/best.pt
test -f /opt/sparseunet4d_deploy/model/config.yaml
```

## 5. Validate the container before starting the robot

```bash
export SU4D_DEPLOY=/opt/sparseunet4d_deploy

docker run --rm --gpus all --network host \
  -v "$SU4D_DEPLOY/mos_venv:/opt/mos_venv:ro" \
  -v "$SU4D_DEPLOY/sparseunet4d:/root/sparseunet4d:ro" \
  -v "$SU4D_DEPLOY/model:/opt/model:ro" \
  -e SU4D_BACKEND=me \
  -e PYTHONPATH=/root/sparseunet4d:/opt/ros/humble/lib/python3.10/site-packages \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mos_venv/lib/python3.10/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  sparseunet4d-husky:runtime \
  bash -lc 'source /opt/ros/humble/setup.bash && \
    /opt/mos_venv/bin/python - <<"PY"
import torch
import MinkowskiEngine as ME
import rclpy
from sparseunet4d.models.backend import backend

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("MinkowskiEngine:", ME.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability())
print("Backend:", backend())
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (8, 6)
assert backend() == "me"
print("CONTAINER VERIFICATION PASSED")
PY'
```

Expected versions are PyTorch `1.12.1+cu113`, PyTorch CUDA `11.3`, and
MinkowskiEngine `0.5.4`.

Validate the checkpoint separately:

```bash
docker run --rm --gpus all --network host \
  -v "$SU4D_DEPLOY/mos_venv:/opt/mos_venv:ro" \
  -v "$SU4D_DEPLOY/sparseunet4d:/root/sparseunet4d:ro" \
  -v "$SU4D_DEPLOY/model:/opt/model:ro" \
  -e SU4D_BACKEND=me \
  -e PYTHONPATH=/root/sparseunet4d \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mos_venv/lib/python3.10/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  sparseunet4d-husky:runtime \
  bash -lc 'cd /root/sparseunet4d && /opt/mos_venv/bin/python - <<"PY"
from mos_inference import MOSInference

model = MOSInference(
    "/opt/model/config.yaml",
    "/opt/model/best.pt",
    device="cuda",
)
print("Checkpoint loaded; threshold:", model.threshold)
PY'
```

The tested checkpoint prints threshold `0.9`. Do not override it with `0.92`.
On SemanticKITTI validation sequence 08, this released checkpoint records
**83.66% point-level moving IoU** at that threshold. This dataset result is a
reference for checkpoint verification; it is not an expected Gazebo or
real-Husky accuracy because those domains and LiDAR characteristics differ.

## 6. Start Gazebo and identify topics

Launch the existing Husky simulation on the host, not inside this ML
container:

```bash
source /opt/ros/humble/setup.bash
source ~/husky_ws/install/setup.bash
ros2 launch <your_husky_package> <your_gazebo_launch_file>.launch.py
```

In another host terminal, find and validate the topics:

```bash
source /opt/ros/humble/setup.bash
source ~/husky_ws/install/setup.bash

ros2 topic list | grep -Ei 'point|cloud|lidar|velodyne|odom'
ros2 topic type /velodyne_points
ros2 topic type /odom
ros2 topic hz /velodyne_points
ros2 topic hz /odom
```

The cloud must be `sensor_msgs/msg/PointCloud2`; odometry must be
`nav_msgs/msg/Odometry`. Replace the example topic names below with the actual
topics.

Odometry must represent the LiDAR pose, or a base pose with the correct rigid
LiDAR extrinsic applied. Incorrect frames or timestamps make static walls look
moving.

## 7. Run the Gazebo node

The container uses host networking so it joins the host's ROS 2 DDS domain.
Use the same `ROS_DOMAIN_ID` as Gazebo.

```bash
export SU4D_DEPLOY=/opt/sparseunet4d_deploy
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export CLOUD_TOPIC=/velodyne_points
export ODOM_TOPIC=/odom

docker run --rm --name sparseunet4d-mos \
  --gpus all --network host --ipc host \
  -v "$SU4D_DEPLOY/mos_venv:/opt/mos_venv:ro" \
  -v "$SU4D_DEPLOY/sparseunet4d:/root/sparseunet4d:ro" \
  -v "$SU4D_DEPLOY/model:/opt/model:ro" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e SU4D_BACKEND=me \
  -e OMP_NUM_THREADS=6 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e PYTHONPATH=/root/sparseunet4d:/opt/ros/humble/lib/python3.10/site-packages \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mos_venv/lib/python3.10/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  sparseunet4d-husky:runtime \
  bash -lc "source /opt/ros/humble/setup.bash && \
    cd /root/sparseunet4d && \
    exec /opt/mos_venv/bin/python mos_node.py --ros-args \
      -p config:=/opt/model/config.yaml \
      -p ckpt:=/opt/model/best.pt \
      -p device:=cuda \
      -p propagate:=false \
      -p pipeline:=true \
      -p use_sim_time:=true \
      -r /sparseunet4d_mos/points:=$CLOUD_TOPIC \
      -r /sparseunet4d_mos/odom:=$ODOM_TOPIC"
```

`pipeline:=true` keeps the newest scan and drops stale queued scans if the
model cannot keep up. The five-frame offsets are `[0, 1, 2, 4, 8]`; the first
eight scans build temporal history.

Do not add `SU4D_THRESHOLD` unless intentionally changing the operating point.
The checkpoint's stored value is `0.9`.

## 8. Confirm and visualize Gazebo output

On the host:

```bash
source /opt/ros/humble/setup.bash
ros2 node list
ros2 topic hz /sparseunet4d_mos/points_moving
ros2 topic info /sparseunet4d_mos/points_labeled -v
ros2 topic info /sparseunet4d_mos/points_moving -v
```

Open RViz on the host:

```bash
rviz2 --ros-args -p use_sim_time:=true
```

Add the input cloud and `/sparseunet4d_mos/points_moving` as `PointCloud2`
displays. Set the fixed frame to a valid `odom`, `map`, or `base_link` frame.

## 9. Run on the real Husky

Use the same container command with:

1. the real LiDAR and localization topics;
2. `use_sim_time:=false`;
3. an operator ready at the emergency stop.

Change this parameter in the command:

```text
-p use_sim_time:=false
```

Start stationary, verify that static structures remain static, and then drive
slowly in a controlled area. Wheel odometry alone may be insufficient; use the
robot's lidar-inertial localization or another accurate pose source.

If no suitable odometry topic exists, KISS-ICP must be installed inside a new
image or portable environment and tested before enabling `use_kiss_icp:=true`.
Do not install it into the host ROS Python.

## 10. Memory and throughput

Start with the original checkpoint configuration (`point_range: 51.2`) so the
robot uses the same input definition as evaluation. If a 4 GB GPU actually
runs out of memory, make a separate deployment configuration:

```bash
cd /opt/sparseunet4d_deploy/sparseunet4d
cp /opt/sparseunet4d_deploy/model/config.yaml \
   configs/deploy_husky_3050ti.yaml
sed -i 's/point_range: 51.2/point_range: 25.6/' \
  configs/deploy_husky_3050ti.yaml
```

Then change the `config:=` path in the run command. Record this change because
it alters the input range.

The controlled SemanticKITTI benchmark measured:

- RTX 3090: approximately `493 ms/scan` end-to-end (`2.0 Hz`);
- network/sparse-tensor stage: approximately `189 ms` (`5.3 Hz`);
- peak allocated VRAM: `1,027 MB`.

Robot/Gazebo clouds may contain fewer points and run faster. `ros2 topic hz`
measures publication frequency, not individual inference latency; use the
latency printed by `mos_node.py` for robot measurements.

## 11. Troubleshooting

### Container cannot see the GPU

```bash
docker run --rm --gpus all \
  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

If this fails, repair the host NVIDIA Container Toolkit configuration before
debugging SparseUNet4D.

### `No module named MinkowskiEngine`

Confirm both the environment mount and interpreter:

```bash
docker run --rm --gpus all \
  -v /opt/sparseunet4d_deploy/mos_venv:/opt/mos_venv:ro \
  sparseunet4d-husky:runtime \
  /opt/mos_venv/bin/python -c 'import MinkowskiEngine; print("ME OK")'
```

### `libcusparse.so.11` is missing

Use the documented CUDA runtime image. Do not replace it with plain
`ubuntu:22.04`.

### Backend is `mock`

The container command must include `-e SU4D_BACKEND=me`.

### ROS topics are invisible

Confirm `--network host`, matching `ROS_DOMAIN_ID`, and compatible ROS
middleware on host and container. Then run:

```bash
ros2 topic list
docker exec sparseunet4d-mos bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic list'
```

### Static structures are classified as moving

Check, in order:

1. LiDAR and odometry timestamps;
2. odometry frame convention;
3. LiDAR-to-base extrinsic;
4. pose accuracy during turns;
5. only then, the moving threshold.

### Inference is slower than expected

Check point count, five-frame voxel count, GPU utilization, CPU load, and the
node's printed latency. Keep `pipeline:=true`. Closing RViz may improve timing,
but it does not change model accuracy.

## 12. Save a reproducibility record

After a successful run, save:

```bash
mkdir -p /opt/sparseunet4d_deploy/run_record

git -C /opt/sparseunet4d_deploy/sparseunet4d rev-parse HEAD \
  > /opt/sparseunet4d_deploy/run_record/git_commit.txt

sha256sum \
  /opt/sparseunet4d_deploy/model/config.yaml \
  /opt/sparseunet4d_deploy/model/best.pt \
  > /opt/sparseunet4d_deploy/run_record/model_sha256.txt

nvidia-smi > /opt/sparseunet4d_deploy/run_record/nvidia_smi.txt
docker image inspect sparseunet4d-husky:runtime \
  > /opt/sparseunet4d_deploy/run_record/docker_image.json
ros2 topic list -t > /opt/sparseunet4d_deploy/run_record/ros_topics.txt
```

Also record the Husky hardware, LiDAR model and rate, average points per scan,
pose source, topic names, ROS domain, simulation world, threshold, point range,
and whether simulation time was enabled.

## Minimal checklist

- [ ] Host `nvidia-smi` works.
- [ ] Docker and NVIDIA Container Toolkit can expose the GPU.
- [ ] GPU compute capability is `sm_86` for the supplied environment.
- [ ] Runtime image is built, exported, checksum-verified, and loaded.
- [ ] `mos_venv`, repository, checkpoint, and configuration are present.
- [ ] Container verification reports backend `me` and threshold `0.9`.
- [ ] LiDAR and odometry topics have the correct ROS message types.
- [ ] Gazebo uses `use_sim_time:=true`; the real robot uses `false`.
- [ ] `pipeline:=true` and `propagate:=false` are set.
- [ ] Output topics are visible and static structures remain static.
- [ ] Reproducibility metadata is saved.
