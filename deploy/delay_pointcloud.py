#!/usr/bin/env python3
"""Delay a PointCloud2 stream for latency-aligned RViz visualization only."""
from collections import deque
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class PointCloudDelay(Node):
    def __init__(self):
        super().__init__("pointcloud_visualization_delay")
        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("output_topic", "/velodyne_points_delayed")
        self.declare_parameter("delay_sec", 0.65)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        delay_sec = float(self.get_parameter("delay_sec").value)
        if delay_sec < 0.0:
            raise ValueError("delay_sec must be non-negative")

        self.delay_ns = int(delay_sec * 1e9)
        # About 20 seconds of capacity at 10 Hz; normal occupancy is 7 scans.
        self.queue = deque(maxlen=200)
        self.publisher = self.create_publisher(
            PointCloud2, output_topic, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.on_cloud, qos_profile_sensor_data)
        self.timer = self.create_timer(0.005, self.release_ready)
        self.get_logger().info(
            f"delaying {input_topic} by {delay_sec:.3f} s -> {output_topic}")

    def on_cloud(self, msg):
        self.queue.append((time.monotonic_ns() + self.delay_ns, msg))

    def release_ready(self):
        now = time.monotonic_ns()
        while self.queue and self.queue[0][0] <= now:
            _, msg = self.queue.popleft()
            # Keep the original header timestamp. The delayed raw cloud and
            # the MOS result therefore use the same capture-time TF lookup.
            self.publisher.publish(msg)


def main():
    rclpy.init()
    node = PointCloudDelay()
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
