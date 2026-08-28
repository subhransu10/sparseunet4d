#!/usr/bin/env python3
"""Record one SparseUNet4D robot trial as frame CSV + JSON summary.

Run on the same PC as ``mos_node.py``.  The raw and labeled clouds are matched
by timestamp, so latency does not require synchronized clocks between the
Husky and the inference PC.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2


def stamp_ns(msg):
    return int(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def mean(values):
    return float(np.mean(values)) if values else None


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "trial"


def moving_clusters(xyz, tolerance, min_points):
    """Fast dependency-free voxel connected components for moving points."""
    if len(xyz) < min_points:
        return 0
    cells, counts = np.unique(
        np.floor(xyz / tolerance).astype(np.int32), axis=0, return_counts=True)
    count_by_cell = {tuple(c): int(n) for c, n in zip(cells, counts)}
    unseen = set(count_by_cell)
    clusters = 0
    neighbours = [(x, y, z) for x in (-1, 0, 1)
                  for y in (-1, 0, 1) for z in (-1, 0, 1)
                  if (x, y, z) != (0, 0, 0)]
    while unseen:
        seed = unseen.pop()
        queue = [seed]
        points = 0
        while queue:
            cell = queue.pop()
            points += count_by_cell[cell]
            for dx, dy, dz in neighbours:
                nxt = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
        if points >= min_points:
            clusters += 1
    return clusters


def model_rss_mib():
    total_kib = 0
    me = os.getpid()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int(proc.name) == me:
                continue
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
            if b"mos_node.py" not in cmd:
                continue
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                UnicodeDecodeError, ValueError):
            continue
    return total_kib / 1024.0 if total_kib else None


def gpu_used_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2)
        values = [float(x.strip()) for x in result.stdout.splitlines()
                  if x.strip()]
        return max(values) if values else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


class TrialRecorder(Node):
    def __init__(self, args):
        super().__init__("sparseunet4d_trial_recorder")
        self.args = args
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
        output_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PointCloud2, args.raw_topic,
                                 self.on_raw, sensor_qos)
        self.create_subscription(PointCloud2, args.labeled_topic,
                                 self.on_labeled, output_qos)
        self.create_subscription(String, args.metrics_topic,
                                 self.on_metrics, 20)
        self.raw_arrivals = OrderedDict()
        self.metrics = OrderedDict()
        self.rows_by_stamp = {}
        self.rows = []
        self.first_output = None
        self.record_start = None
        self.last_memory_sample = 0.0
        self.model_rss_samples = []
        self.gpu_samples = []
        self.done = False

    @staticmethod
    def _trim(cache, maximum=1000):
        while len(cache) > maximum:
            cache.popitem(last=False)

    def on_raw(self, msg):
        self.raw_arrivals[stamp_ns(msg)] = (time.perf_counter(),
                                            int(msg.width * msg.height))
        self._trim(self.raw_arrivals)

    def on_metrics(self, msg):
        try:
            metric = json.loads(msg.data)
            stamp = int(metric["stamp_ns"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return
        self.metrics[stamp] = metric
        self._trim(self.metrics)
        row = self.rows_by_stamp.get(stamp)
        if row is not None:
            self._apply_metric(row, metric)

    @staticmethod
    def _apply_metric(row, metric):
        row["model_latency_ms"] = metric.get("latency_ms", "")
        row["active_4d_voxels"] = metric.get("active_4d_voxels", "")
        row["pose_mode"] = metric.get("pose_mode", "")

    def _sample_memory(self, now):
        if now - self.last_memory_sample < self.args.memory_interval:
            return
        self.last_memory_sample = now
        rss = model_rss_mib()
        gpu = gpu_used_mib()
        if rss is not None:
            self.model_rss_samples.append(rss)
        if gpu is not None:
            self.gpu_samples.append(gpu)

    def on_labeled(self, msg):
        now = time.perf_counter()
        if self.first_output is None:
            self.first_output = now
            self.get_logger().info(
                f"output found; warming up for {self.args.warmup:g} s")
        if now - self.first_output < self.args.warmup:
            return
        if self.record_start is None:
            self.record_start = now
            self.get_logger().info(
                f"RECORDING {self.args.trial} for {self.args.duration:g} s")

        data = np.asarray(pc2.read_points(
            msg, field_names=("x", "y", "z", "moving", "moving_prob"),
            skip_nans=True))
        if data.dtype.names:
            xyz = np.column_stack((data["x"], data["y"], data["z"]))
            moving = np.asarray(data["moving"])
            prob = np.asarray(data["moving_prob"])
        else:
            xyz, moving, prob = data[:, :3], data[:, 3], data[:, 4]

        valid = moving != 255
        moving_mask = moving == 1
        n_total = int(len(moving))
        n_valid = int(np.count_nonzero(valid))
        n_moving = int(np.count_nonzero(moving_mask))
        ratio = n_moving / max(n_valid, 1)
        if n_valid:
            voxels_3d = len(np.unique(
                np.floor(xyz[valid] / self.args.voxel_size).astype(np.int32),
                axis=0))
        else:
            voxels_3d = 0
        clusters = moving_clusters(xyz[moving_mask],
                                   self.args.cluster_tolerance,
                                   self.args.cluster_min_points)
        stamp = stamp_ns(msg)
        raw = self.raw_arrivals.get(stamp)
        transport_latency = ((now - raw[0]) * 1e3 if raw else "")
        row = {
            "stamp_ns": stamp,
            "elapsed_s": now - self.record_start,
            "input_points": raw[1] if raw else n_total,
            "in_range_points": n_valid,
            "occupied_3d_voxels": int(voxels_3d),
            "active_4d_voxels": "",
            "moving_points": n_moving,
            "moving_ratio": ratio,
            "max_moving_probability": float(prob.max()) if len(prob) else 0.0,
            "moving_clusters": clusters,
            "cluster_detected": int(clusters > 0),
            "matched_input_output_latency_ms": transport_latency,
            "model_latency_ms": "",
            "pose_mode": "",
        }
        metric = self.metrics.get(stamp)
        if metric:
            self._apply_metric(row, metric)
        self.rows.append(row)
        self.rows_by_stamp[stamp] = row
        self._sample_memory(now)
        if now - self.record_start >= self.args.duration:
            self.done = True


def numeric(rows, key):
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def write_results(node, args):
    if not node.rows:
        raise RuntimeError("no labeled frames were recorded")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + safe_name(args.trial)
    csv_path = out_dir / f"{run_id}_frames.csv"
    json_path = out_dir / f"{run_id}_summary.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(node.rows[0]))
        writer.writeheader()
        writer.writerows(node.rows)

    elapsed = numeric(node.rows, "elapsed_s")
    intervals = np.diff(elapsed).tolist() if len(elapsed) > 1 else []
    moving_ratios = numeric(node.rows, "moving_ratio")
    clusters = numeric(node.rows, "moving_clusters")
    detected = numeric(node.rows, "cluster_detected")
    model_latency = numeric(node.rows, "model_latency_ms")
    matched_latency = numeric(node.rows, "matched_input_output_latency_ms")
    active_4d = numeric(node.rows, "active_4d_voxels")
    pose_modes = sorted({str(r["pose_mode"]) for r in node.rows
                         if r.get("pose_mode")})
    summary = {
        "trial": args.trial,
        "expected": args.expected,
        "condition": {
            "point_range_m": args.range_m,
            "robot_motion": args.robot_motion,
            "slam": args.slam,
            "notes": args.notes,
            "pose_mode": pose_modes[0] if len(pose_modes) == 1 else pose_modes,
        },
        "system": {
            "host": platform.node(),
            "machine": platform.machine(),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        },
        "topics": {
            "raw": args.raw_topic,
            "labeled": args.labeled_topic,
            "metrics": args.metrics_topic,
        },
        "frames": len(node.rows),
        "duration_s": elapsed[-1] if elapsed else 0.0,
        "output_rate_hz": (1.0 / mean(intervals)) if intervals else None,
        "points_per_scan_mean": mean(numeric(node.rows, "input_points")),
        "in_range_points_mean": mean(numeric(node.rows, "in_range_points")),
        "occupied_3d_voxels_mean": mean(
            numeric(node.rows, "occupied_3d_voxels")),
        "active_4d_voxels_mean": mean(active_4d),
        "model_latency_ms": {
            "mean": mean(model_latency),
            "median": percentile(model_latency, 50),
            "p95": percentile(model_latency, 95),
        },
        "matched_input_output_latency_ms": {
            "mean": mean(matched_latency),
            "median": percentile(matched_latency, 50),
            "p95": percentile(matched_latency, 95),
        },
        "moving_point_ratio_percent_mean": 100.0 * mean(moving_ratios),
        "moving_clusters_per_frame_mean": mean(clusters),
        "cluster_positive_frame_rate_percent": 100.0 * mean(detected),
        "detection_rate_percent": (100.0 * mean(detected)
                                   if args.expected == "moving" else None),
        "static_false_positive_frame_rate_percent": (
            100.0 * mean(detected) if args.expected == "static" else None),
        "peak_model_rss_mib": max(node.model_rss_samples, default=None),
        "peak_total_gpu_memory_used_mib": max(node.gpu_samples, default=None),
        "cluster_definition": {
            "tolerance_m": args.cluster_tolerance,
            "minimum_points": args.cluster_min_points,
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    master = out_dir / "trials.csv"
    flat = {
        "trial": args.trial,
        "expected": args.expected,
        "range_m": args.range_m,
        "robot_motion": args.robot_motion,
        "slam": args.slam,
        "pose_mode": summary["condition"]["pose_mode"],
        "frames": summary["frames"],
        "rate_hz": summary["output_rate_hz"],
        "latency_mean_ms": summary["model_latency_ms"]["mean"],
        "latency_p95_ms": summary["model_latency_ms"]["p95"],
        "points_mean": summary["points_per_scan_mean"],
        "active_4d_voxels_mean": summary["active_4d_voxels_mean"],
        "moving_ratio_percent": summary["moving_point_ratio_percent_mean"],
        "clusters_per_frame": summary["moving_clusters_per_frame_mean"],
        "positive_frame_rate_percent": summary[
            "cluster_positive_frame_rate_percent"],
        "peak_model_rss_mib": summary["peak_model_rss_mib"],
        "peak_gpu_memory_mib": summary["peak_total_gpu_memory_used_mib"],
        "summary_json": str(json_path),
    }
    write_header = not master.exists()
    with master.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    print("\n================ ROBOT TRIAL ================")
    print(f"trial:                 {args.trial}")
    print(f"frames:                {summary['frames']}")
    print(f"output rate:           {summary['output_rate_hz']:.2f} Hz")
    if summary["model_latency_ms"]["mean"] is not None:
        print(f"model latency:         {summary['model_latency_ms']['mean']:.1f} ms "
              f"(p95 {summary['model_latency_ms']['p95']:.1f})")
    print(f"moving-point ratio:    "
          f"{summary['moving_point_ratio_percent_mean']:.3f}%")
    print(f"positive frames:       "
          f"{summary['cluster_positive_frame_rate_percent']:.1f}%")
    print(f"clusters/frame:        "
          f"{summary['moving_clusters_per_frame_mean']:.3f}")
    print(f"frame CSV:             {csv_path}")
    print(f"summary JSON:          {json_path}")
    print(f"all-trials table:      {master}")
    print("=============================================\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Record a paper-ready SparseUNet4D robot trial")
    ap.add_argument("--trial", required=True, help="short unique trial name")
    ap.add_argument("--expected", choices=("static", "moving", "unspecified"),
                    default="unspecified")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--range-m", type=float, default=None)
    ap.add_argument("--robot-motion",
                    choices=("stationary", "translation", "turning", "mixed",
                             "unspecified"), default="unspecified")
    ap.add_argument("--slam", choices=("on", "off", "unspecified"),
                    default="unspecified")
    ap.add_argument("--notes", default="")
    ap.add_argument("--raw-topic", default="/velodyne_points")
    ap.add_argument("--labeled-topic",
                    default="/sparseunet4d_mos/points_labeled")
    ap.add_argument("--metrics-topic",
                    default="/sparseunet4d_mos/metrics")
    ap.add_argument("--voxel-size", type=float, default=0.1)
    ap.add_argument("--cluster-tolerance", type=float, default=0.5)
    ap.add_argument("--cluster-min-points", type=int, default=5)
    ap.add_argument("--memory-interval", type=float, default=2.0)
    ap.add_argument("--startup-timeout", type=float, default=45.0)
    ap.add_argument("--output-dir", default="results/robot_trials")
    return ap.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = TrialRecorder(args)
    started = time.monotonic()
    node.get_logger().info(
        f"waiting for Reliable output on {args.labeled_topic}")
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (node.first_output is None
                    and time.monotonic() - started > args.startup_timeout):
                publishers = node.count_publishers(args.labeled_topic)
                raise RuntimeError(
                    f"no messages on {args.labeled_topic} after "
                    f"{args.startup_timeout:g} s (discovered publishers: "
                    f"{publishers}). Keep mos_node.py running in another "
                    "terminal and verify ROS_DOMAIN_ID/RMW_IMPLEMENTATION.")
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        print("\nStopped early; saving the partial trial.")
    finally:
        try:
            if node.rows:
                write_results(node, args)
            else:
                print("No trial file written because no labeled frames were "
                      "received.")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
