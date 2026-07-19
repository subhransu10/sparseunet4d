"""ROS2 node: streaming LiDAR moving-object segmentation with SparseUNet4D.

Subscribes:
  ~/points   sensor_msgs/PointCloud2   raw scan (x, y, z, intensity)
  ~/odom     nav_msgs/Odometry         sensor pose in a fixed frame [optional]

Publishes:
  ~/points_labeled  PointCloud2  input cloud + fields:
                      moving (uint8: 0 static, 1 moving, 255 = out of range)
                      moving_prob (float32)
  ~/points_moving   PointCloud2  moving points only (convenience, e.g. for
                                 dynamic-obstacle costmap layers)

Pose source:
  odom topic if present, else embedded KISS-ICP (pip install kiss-icp).
  MEASURED REQUIREMENT (val-08 drift sweep): relative pose error over the
  4-frame (~0.3 s) window must stay under ~5 cm. At 1-3 cm (typical LiDAR
  odometry) the IoU cost is <= 0.6 points; at 8 cm it is 2.7 points; at 11 cm
  it is 8.5 points and falls off a cliff beyond that. Wheel odometry alone
  will NOT meet this. The node warns if consecutive poses imply motion
  inconsistent with the LiDAR rate.

Latency: preprocessing (voxelize + spherical projections + residuals) is
CPU-heavy and runs in a worker thread; inference runs on GPU. With
`~pipeline:=true` the node processes the newest scan and DROPS older queued
scans rather than falling behind (correct behaviour for a robot: fresh
predictions beat complete ones).

Run (current best model — strided 5-frame window [1,2,4,8]):
  SU4D_BACKEND=me PYTHONPATH=$HOME/MinkowskiEngine:$HOME/sparseunet4d \
  ros2 run <pkg> mos_node --ros-args \
    -p config:=$HOME/sparseunet4d/configs/residual_inject.yaml \
    -p ckpt:=$HOME/sparseunet4d/runs/residual_inject2/best.pt \
    -p propagate:=true \
    -r ~/points:=/velodyne_points -r ~/odom:=/odometry/lidar

  For deployment under drifty online odometry, use the drift-robust checkpoint
  instead:  -p ckpt:=$HOME/sparseunet4d/runs/consistency_ft/best.pt

WARM-UP: the widest offset is 8, so the first ~8 scans (~0.8 s) produce
partial-window predictions (missing offsets contribute zero residual) before
the buffer is full. This is by design and matches early-in-sequence training.

DOMAIN NOTE: the model is trained on 64-beam SemanticKITTI (HDL-64E). A Husky
with a 16/32-beam LiDAR has very different point density and residual
statistics -- expect a domain gap. For a real demo, either use a 64-beam
sensor, or fine-tune on a little labelled robot data. The pipeline is correct
regardless; accuracy transfer is the open question.
"""
from __future__ import annotations
import os, sys, threading, time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from mos_inference import MOSInference          # noqa: E402

MAX_REL_T = 0.05          # m, per-window budget from the drift sweep


def quat_to_R(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


class MOSNode(Node):
    def __init__(self):
        super().__init__("sparseunet4d_mos")
        self.declare_parameter("config", "")
        self.declare_parameter("ckpt", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("propagate", True)
        self.declare_parameter("use_kiss_icp", False)
        self.declare_parameter("pipeline", True)
        cfg = self.get_parameter("config").value
        ckpt = self.get_parameter("ckpt").value
        assert cfg and ckpt, "config and ckpt parameters are required"
        self.pipeline = self.get_parameter("pipeline").value

        self.get_logger().info("loading SparseUNet4D...")
        self.mos = MOSInference(cfg, ckpt,
                                device=self.get_parameter("device").value,
                                propagate=self.get_parameter("propagate").value)

        self.icp = None
        if self.get_parameter("use_kiss_icp").value:
            from kiss_icp.pipeline import OdometryPipeline  # noqa
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
            self.icp = KissICP(config=load_config(None))
            self.get_logger().info("using embedded KISS-ICP for poses")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub_pc = self.create_subscription(
            PointCloud2, "~/points", self.on_cloud, qos)
        self.sub_odom = self.create_subscription(
            Odometry, "~/odom", self.on_odom, qos)
        self.pub_all = self.create_publisher(PointCloud2, "~/points_labeled", 1)
        self.pub_mov = self.create_publisher(PointCloud2, "~/points_moving", 1)

        self._odom = None                 # latest (stamp_ns, T 4x4)
        self._pending = None              # newest unprocessed (msg, T)
        self._lock = threading.Lock()
        self._prev_T = None
        self._busy = False
        self._t_last_log = time.time()
        self._lat = []
        if self.pipeline:
            threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info("ready.")

    # ---------------- callbacks ------------------------------------------
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        T = np.eye(4)
        T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._odom = (stamp, T)

    def on_cloud(self, msg: PointCloud2):
        T = self._pose_for(msg)
        if T is None:
            self.get_logger().warn("no pose available yet; dropping scan",
                                   throttle_duration_sec=2.0)
            return
        if self.pipeline:
            with self._lock:
                self._pending = (msg, T)   # newest wins; stale scans dropped
        else:
            self._process(msg, T)

    # ---------------- pose ------------------------------------------------
    def _pose_for(self, msg):
        if self.icp is not None:
            xyz = self._read_xyzi(msg)[:, :3].astype(np.float64)
            self.icp.register_frame(xyz, np.zeros(len(xyz)))
            T = np.asarray(self.icp.last_pose)
        elif self._odom is not None:
            T = self._odom[1]
        else:
            return None
        if self._prev_T is not None:
            d = np.linalg.norm(np.linalg.inv(self._prev_T)[:3, :3]
                               @ (T[:3, 3] - self._prev_T[:3, 3]))
            if d > 5.0:      # >5 m between consecutive scans at ~10 Hz = 180 km/h
                self.get_logger().warn(
                    f"implausible inter-scan motion {d:.2f} m -- check odometry "
                    "frame/rate; MOS needs <5 cm RELATIVE error over 4 frames")
        self._prev_T = T
        return T

    # ---------------- worker ---------------------------------------------
    def _worker(self):
        while rclpy.ok():
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                time.sleep(0.002)
                continue
            self._process(*job)

    def _process(self, msg, T):
        t0 = time.time()
        scan = self._read_xyzi(msg)
        labels, probs = self.mos.push(scan, T)
        self._publish(msg, scan, labels, probs)
        self._lat.append(time.time() - t0)
        if time.time() - self._t_last_log > 5.0:
            l = np.array(self._lat) * 1e3
            self.get_logger().info(
                f"latency mean {l.mean():.1f} ms (p95 {np.percentile(l,95):.1f}) "
                f"-> {1000/max(l.mean(),1e-6):.1f} Hz")
            self._lat.clear(); self._t_last_log = time.time()

    # ---------------- IO ---------------------------------------------------
    @staticmethod
    def _read_xyzi(msg):
        a = pc2.read_points_numpy(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=False)
        return np.asarray(a, np.float32).reshape(-1, 4)

    def _publish(self, msg, scan, labels, probs):
        mv = labels.copy()
        mv[mv < 0] = 255                       # out-of-range clip
        rec = np.zeros(len(scan), dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("intensity", np.float32), ("moving", np.uint8),
            ("moving_prob", np.float32)])
        rec["x"], rec["y"], rec["z"] = scan[:, 0], scan[:, 1], scan[:, 2]
        rec["intensity"] = scan[:, 3]
        rec["moving"] = mv.astype(np.uint8)
        rec["moving_prob"] = probs
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="moving", offset=16,
                       datatype=PointField.UINT8, count=1),
            PointField(name="moving_prob", offset=17,
                       datatype=PointField.FLOAT32, count=1)]
        self.pub_all.publish(
            pc2.create_cloud(msg.header, fields, rec))
        m = labels == 1
        if m.any():
            self.pub_mov.publish(
                pc2.create_cloud(msg.header, fields[:4], scan[m]))


def main():
    rclpy.init()
    node = MOSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()