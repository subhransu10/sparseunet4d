"""Replay a KITTI odometry sequence as live ROS 2 topics for the MOS node.

Publishes, at --rate Hz:
  /points  sensor_msgs/PointCloud2  (velodyne frame, x y z intensity)
  /odom    nav_msgs/Odometry        (map -> velodyne, GROUND-TRUTH pose)
  TF       map -> velodyne          (so RViz Fixed Frame = map is world-fixed)

This is the exact data the model was trained/evaluated on: dense 64-beam +
perfect poses -> the cleanest possible live demo (no KISS-ICP needed).

Usage:
  source /opt/ros/humble/setup.bash && source ~/mos_venv/bin/activate
  python kitti_replay.py --seq-dir /path/to/sequences/08 --rate 10 --loop
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets.poses import GTPoseProvider   # velodyne-in-world poses


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


class KittiReplay(Node):
    def __init__(self, seq_dir, rate, loop):
        super().__init__("kitti_replay")
        self.velo = os.path.join(seq_dir, "velodyne")
        self.files = sorted(f for f in os.listdir(self.velo) if f.endswith(".bin"))
        self.poses = GTPoseProvider(seq_dir).poses          # (F, 4, 4)
        assert len(self.poses) >= len(self.files), "poses/scan count mismatch"
        self.pub_pc = self.create_publisher(PointCloud2, "/points", 5)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 5)
        self.br = TransformBroadcaster(self)
        self.i, self.loop = 0, loop
        self.fields = [PointField(name=n, offset=4 * k,
                                  datatype=PointField.FLOAT32, count=1)
                       for k, n in enumerate(("x", "y", "z", "intensity"))]
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"replaying {len(self.files)} scans at {rate} Hz")

    def tick(self):
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                self.get_logger().info("sequence finished")
                rclpy.shutdown()
                return
        scan = np.fromfile(os.path.join(self.velo, self.files[self.i]),
                           np.float32).reshape(-1, 4)
        now = self.get_clock().now().to_msg()

        hdr = Header(); hdr.stamp = now; hdr.frame_id = "velodyne"
        pc = PointCloud2(); pc.header = hdr
        pc.height = 1; pc.width = scan.shape[0]; pc.fields = self.fields
        pc.is_bigendian = False; pc.point_step = 16
        pc.row_step = 16 * scan.shape[0]; pc.is_dense = True
        pc.data = scan.tobytes()                            # xyzi is already the wire layout
        self.pub_pc.publish(pc)

        T = self.poses[self.i]
        qx, qy, qz, qw = R_to_quat(T[:3, :3])
        od = Odometry(); od.header.stamp = now
        od.header.frame_id = "map"; od.child_frame_id = "velodyne"
        od.pose.pose.position.x = float(T[0, 3])
        od.pose.pose.position.y = float(T[1, 3])
        od.pose.pose.position.z = float(T[2, 3])
        od.pose.pose.orientation.x = qx; od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz; od.pose.pose.orientation.w = qw
        self.pub_odom.publish(od)

        tf = TransformStamped(); tf.header.stamp = now
        tf.header.frame_id = "map"; tf.child_frame_id = "velodyne"
        tf.transform.translation.x = float(T[0, 3])
        tf.transform.translation.y = float(T[1, 3])
        tf.transform.translation.z = float(T[2, 3])
        tf.transform.rotation = od.pose.pose.orientation
        self.br.sendTransform(tf)
        self.i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help=".../sequences/08")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    node = KittiReplay(args.seq_dir, args.rate, args.loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
