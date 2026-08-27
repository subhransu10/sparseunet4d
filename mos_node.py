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
`~pipeline:=true` the node publishes the newest scan and drops older queued
output jobs rather than falling behind. Every input scan is still retained in
the temporal history, so offsets [1,2,4,8] stay tied to the LiDAR rate.

Run (released model — strided 5-frame window [1,2,4,8]):
  SU4D_BACKEND=me PYTHONPATH=$HOME/MinkowskiEngine:$HOME/sparseunet4d \
  ros2 run <pkg> mos_node --ros-args \
    -p config:=$HOME/sparseunet4d/configs/pretrained_semantickitti.yaml \
    -p ckpt:=$HOME/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt \
    -p propagate:=false \
    -r /sparseunet4d_mos/points:=/velodyne_points \
    -r /sparseunet4d_mos/odom:=/odometry/lidar

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
    if not np.isfinite(n) or n < 1e-8:
        return np.eye(3)          # degenerate/uninitialised quaternion
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
        self.declare_parameter("intensity_scale", 1.0)
        self.declare_parameter("projection_height", 64)
        self.declare_parameter("projection_width", 2048)
        self.declare_parameter("fov_up_deg", 3.0)
        self.declare_parameter("fov_down_deg", -25.0)
        cfg = self.get_parameter("config").value
        ckpt = self.get_parameter("ckpt").value
        assert cfg and ckpt, "config and ckpt parameters are required"
        self.pipeline = self.get_parameter("pipeline").value
        self.intensity_scale = float(
            self.get_parameter("intensity_scale").value)
        if not np.isfinite(self.intensity_scale) or self.intensity_scale <= 0:
            raise ValueError("intensity_scale must be finite and greater than zero")

        self.get_logger().info("loading SparseUNet4D...")
        self.mos = MOSInference(cfg, ckpt,
                                device=self.get_parameter("device").value,
                                propagate=self.get_parameter("propagate").value,
                                projection_height=self.get_parameter(
                                    "projection_height").value,
                                projection_width=self.get_parameter(
                                    "projection_width").value,
                                fov_up_deg=self.get_parameter("fov_up_deg").value,
                                fov_down_deg=self.get_parameter(
                                    "fov_down_deg").value)
        self.get_logger().info(
            f"sensor adaptation: intensity / {self.intensity_scale:g}, "
            f"projection {self.mos.projection_height}x"
            f"{self.mos.projection_width}, vertical FOV "
            f"[{self.mos.fov_down_deg:g}, {self.mos.fov_up_deg:g}] deg")

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
        self._odom_buf = []               # [(stamp_ns, pos3, quat4)] time-sync
        self._scan_buf = []               # every input scan, newest first
        self._pending = None              # newest output job; stale jobs dropped
        self._lock = threading.Lock()
        self._prev_T = None
        self._busy = False
        self._t_last_log = time.time()
        self._lat = []
        self._prob_max = []
        self._moving_count = []
        self._logged_input = False
        if self.pipeline:
            threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info("ready.")

    # ---------------- callbacks ------------------------------------------
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._odom_buf.append((stamp, np.array([p.x, p.y, p.z]),
                               np.array([q.x, q.y, q.z, q.w])))
        if len(self._odom_buf) > 300:
            self._odom_buf.pop(0)
        T = np.eye(4)
        T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        self._odom = (stamp, T)

    def on_cloud(self, msg: PointCloud2):
        scan = self._read_xyzi(msg)
        if not len(scan):
            self.get_logger().warn("empty finite point cloud; dropping scan",
                                   throttle_duration_sec=2.0)
            return
        T = self._pose_for(msg, scan)
        if T is None:
            self.get_logger().warn("no pose available yet; dropping scan",
                                   throttle_duration_sec=2.0)
            return
        model_scan = scan.copy()
        model_scan[:, 3] /= self.intensity_scale
        if not self._logged_input:
            raw = scan[:, 3]
            pct = np.percentile(raw, [0, 50, 95, 100])
            self.get_logger().info(
                "input points=%d raw intensity min/median/p95/max="
                "%.3g/%.3g/%.3g/%.3g; normalized max=%.3g" %
                (len(scan), *pct, model_scan[:, 3].max()))
            self._logged_input = True

        frame = (model_scan[:, :3], model_scan[:, 3:4],
                 np.asarray(T, np.float64).copy())
        self._scan_buf.insert(0, frame)
        del self._scan_buf[self.mos.max_offset + 1:]
        history = tuple(self._scan_buf)
        if self.pipeline:
            with self._lock:
                self._pending = (msg, scan, history)  # newest output wins
        else:
            self._process(msg, scan, history)

    # ---------------- pose ------------------------------------------------
    @staticmethod
    def _slerp(q0, q1, u):
        d = float(np.dot(q0, q1))
        if d < 0.0:
            q1 = -q1; d = -d
        if d > 0.9995:
            q = q0 + u * (q1 - q0)
            return q / np.linalg.norm(q)
        th = np.arccos(d); si = np.sin(th)
        return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / si

    def _pose_at(self, stamp):
        buf = self._odom_buf
        if not buf:
            return None
        if stamp <= buf[0][0]:
            _, p, q = buf[0]
        elif stamp >= buf[-1][0]:
            _, p, q = buf[-1]
        else:
            p = q = None
            for i in range(1, len(buf)):
                if buf[i][0] >= stamp:
                    s0, p0, q0 = buf[i - 1]
                    s1, p1, q1 = buf[i]
                    u = (stamp - s0) / max(s1 - s0, 1)
                    p = p0 + u * (p1 - p0)
                    q = self._slerp(q0, q1, u)
                    break
        T = np.eye(4)
        T[:3, :3] = quat_to_R(q[0], q[1], q[2], q[3])
        T[:3, 3] = p
        return T

    def _pose_for(self, msg, scan=None):
        if self.icp is not None:
            if scan is None:
                scan = self._read_xyzi(msg)
            xyz = scan[:, :3].astype(np.float64)
            self.icp.register_frame(xyz, np.zeros(len(xyz)))
            T = np.asarray(self.icp.last_pose)
        elif self._odom_buf:
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            T = self._pose_at(stamp)      # pose interpolated to THIS scan's time
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

    def _process(self, msg, scan, history):
        t0 = time.time()
        labels, probs = self.mos.infer_history(history)
        self._publish(msg, scan, labels, probs)
        self._lat.append(time.time() - t0)
        self._prob_max.append(float(probs.max()) if len(probs) else 0.0)
        self._moving_count.append(int(np.count_nonzero(labels == 1)))
        if time.time() - self._t_last_log > 5.0:
            l = np.array(self._lat) * 1e3
            self.get_logger().info(
                f"latency mean {l.mean():.1f} ms (p95 {np.percentile(l,95):.1f}) "
                f"-> {1000/max(l.mean(),1e-6):.1f} Hz; "
                f"prob max {max(self._prob_max):.3g}; "
                f"moving points max {max(self._moving_count)}")
            self._lat.clear()
            self._prob_max.clear()
            self._moving_count.clear()
            self._t_last_log = time.time()

    # ---------------- IO ---------------------------------------------------
    @staticmethod
    def _read_xyzi(msg):
        names = [f.name for f in msg.fields]
        want = ("x", "y", "z", "intensity") if "intensity" in names else ("x", "y", "z")
        a = pc2.read_points(msg, field_names=want, skip_nans=False)
        a = np.asarray(a).reshape(-1)
        n = int(a.shape[0])
        out = np.zeros((n, 4), np.float32)
        out[:, 0] = np.asarray(a["x"], np.float32)
        out[:, 1] = np.asarray(a["y"], np.float32)
        out[:, 2] = np.asarray(a["z"], np.float32)
        if "intensity" in want:
            out[:, 3] = np.asarray(a["intensity"], np.float32)
        return out[np.isfinite(out).all(axis=1)]   # drop no-return NaN/inf rays

    def _publish(self, msg, scan, labels, probs):
        mv = labels.astype(np.int16)
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
        # SIGINT may already have shut down the default context.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
