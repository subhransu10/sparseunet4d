# Husky robot evaluation

This workflow records paper-ready robot measurements without hand annotation.
Run `mos_node.py` normally, then start one command in a second terminal on the
same inference PC. Each trial produces a per-frame CSV, a JSON summary, and one
row in `results/robot_trials/trials.csv`.

## Before each trial

Use the real-robot sensor parameters and ROS domain. The default synchronized
deployment uses timestamp-interpolated odometry:

```bash
export ROS_DOMAIN_ID=30

python mos_node.py --ros-args \
  -p config:="$MODEL_CONFIG" \
  -p ckpt:="$MODEL_CKPT" \
  -p device:=cuda -p propagate:=false -p pipeline:=true \
  -p intensity_scale:=255.0 \
  -p projection_height:=16 -p projection_width:=2048 \
  -p fov_down_deg:=-15.0 -p fov_up_deg:=15.0 \
  -p pose_mode:=interpolated -p use_sim_time:=false \
  -r /sparseunet4d_mos/points:=/velodyne_points \
  -r /sparseunet4d_mos/odom:=/platform/odom
```

Keep that terminal running. In a second ROS-sourced terminal:

```bash
cd ~/sparseunet4d
source ~/mos_venv/bin/activate
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
```

## Three recommended trials

Park the robot and keep the scene static:

```bash
python deploy/record_robot_trial.py \
  --trial static_stationary_26m --expected static --duration 60 \
  --range-m 25.6 --robot-motion stationary --slam off
```

Have one person walk while the robot remains parked:

```bash
python deploy/record_robot_trial.py \
  --trial person_stationary_robot_26m --expected moving --duration 60 \
  --range-m 25.6 --robot-motion stationary --slam off
```

Have the person walk while the Husky drives and turns:

```bash
python deploy/record_robot_trial.py \
  --trial person_moving_robot_26m --expected moving --duration 60 \
  --range-m 25.6 --robot-motion mixed --slam on
```

The script waits five seconds for warm-up, prints `RECORDING`, records for one
minute, and saves everything automatically. A frame counts as a detection when
it contains a connected moving cluster of at least five points. Use the same
cluster settings for every trial.

## Range and SLAM tests

Restart `mos_node.py` with the desired deployment config, then repeat the
recorder command with the matching `--range-m`. Recommended ranges are 15,
25.6, 35 and 51.2 m. For the SLAM comparison, keep the range and scene fixed
and change only `--slam on/off` and whether the SLAM process is running.

The exact active 4D voxel count and model latency come from
`/sparseunet4d_mos/metrics`; the input/output latency is independently matched
using the cloud timestamp, so the robot and PC clocks do not need to agree.

## Odometry synchronization ablation

Run the same route twice. First use `-p pose_mode:=interpolated`. Then restart
the node with:

```bash
-p pose_mode:=latest
```

Record both trials with identical conditions. `latest` is intentionally an
ablation; normal deployment should remain `interpolated`.

## Results

The compact table is:

```bash
column -s, -t results/robot_trials/trials.csv | less -S
```

For the paper, report output rate, model latency, active 4D voxels, moving-point
ratio, clusters per frame, and cluster-positive frame rate. For a static trial,
the last metric is the false-positive frame rate. For a controlled moving-person
trial, it is the frame-level detection rate; state clearly that this is a
trial-level proxy rather than point-wise ground truth.
