#!/usr/bin/env python3
"""Evaluate native causal geometric proposals for SparseUNet4D.

The prediction path is label-free: a frozen SparseUNet4D checkpoint supplies
point motion scores and object clusters, while registered historical scans
provide a visibility-aware range envelope.  A point receives geometric
disagreement only when a historical return exists in its local range-image
neighbourhood; unobserved rays abstain instead of becoming false positives.

Ground truth is read only after all prediction scores for a frame are built and
is used solely to accumulate the official point-level MOS metric.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np
import torch
import yaml
from scipy.ndimage import maximum_filter, minimum_filter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_eval import load_model, threshold_counts, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.residual_features import spherical_project
from sparseunet4d.datasets.semantickitti import _read_scan, _transform


DEFAULT_THRESHOLDS = [
    0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005,
    0.0075, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
    0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85,
    0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
]


def local_range_envelope(range_image: np.ndarray, radius: int = 1):
    """Return local minimum/maximum historical range and visibility.

    Vertical image boundaries are constant-empty while horizontal boundaries
    wrap because LiDAR azimuth is periodic.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    image = np.asarray(range_image, dtype=np.float64)
    size = 2 * int(radius) + 1
    low = minimum_filter(
        image, size=size, mode=("constant", "wrap"), cval=np.inf)
    finite_image = np.where(np.isfinite(image), image, -np.inf)
    high = maximum_filter(
        finite_image, size=size, mode=("constant", "wrap"), cval=-np.inf)
    visible = np.isfinite(low) & np.isfinite(high)
    return low, high, visible


def causal_range_gaps(current_xyz: np.ndarray,
                      historical_scans_in_current: list[np.ndarray],
                      neighborhood_radius: int = 1) -> np.ndarray:
    """Distance outside each historical local range envelope.

    The returned array is ``(N, K)``. NaN means that history had no return in
    the local ray neighbourhood and therefore contributes no evidence.
    """
    current_xyz = np.asarray(current_xyz, dtype=np.float32)
    _, u, v, current_range, current_valid = spherical_project(current_xyz)
    gaps = np.full(
        (len(current_xyz), len(historical_scans_in_current)),
        np.nan, dtype=np.float32)
    for column, historical_xyz in enumerate(historical_scans_in_current):
        history_image, _, _, _, _ = spherical_project(historical_xyz)
        low_image, high_image, visible_image = local_range_envelope(
            history_image, neighborhood_radius)
        visible = visible_image[v, u] & current_valid
        low = low_image[v, u]
        high = high_image[v, u]
        gap = np.maximum(np.maximum(low - current_range,
                                    current_range - high), 0.0)
        gaps[visible, column] = gap[visible].astype(np.float32)
    return gaps


def geometric_disagreement(gaps: np.ndarray, delta_m: float,
                           minimum_history_hits: int):
    """Convert range gaps to a per-point consensus disagreement score."""
    if delta_m < 0:
        raise ValueError("delta_m must be nonnegative")
    if minimum_history_hits < 1:
        raise ValueError("minimum_history_hits must be positive")
    gaps = np.asarray(gaps, dtype=np.float32)
    observed = np.isfinite(gaps)
    hits = observed.sum(axis=1)
    disagreement = ((gaps > float(delta_m)) & observed).sum(axis=1)
    score = np.zeros(len(gaps), dtype=np.float32)
    eligible = hits >= int(minimum_history_hits)
    score[eligible] = disagreement[eligible] / hits[eligible]
    return score, eligible


def top_fraction(values: np.ndarray, fraction: float = 0.25) -> float:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    count = max(1, int(np.ceil(float(fraction) * len(values))))
    start = len(values) - count
    return float(np.partition(values, start)[start:].mean())


def complete_cluster_scores(point_score: np.ndarray,
                            point_cluster: np.ndarray,
                            geometric_score: np.ndarray | None = None,
                            geometric_eligible: np.ndarray | None = None,
                            boost: float = 0.0,
                            geometric_floor: float = 1.0,
                            minimum_coverage: float = 0.25,
                            fraction: float = 0.25) -> np.ndarray:
    """Top-fraction object completion with optional geometric promotion."""
    output = np.asarray(point_score, dtype=np.float32).copy()
    clusters = np.asarray(point_cluster)
    if geometric_score is not None:
        geometric_score = np.asarray(geometric_score, dtype=np.float32)
        geometric_eligible = np.asarray(geometric_eligible, dtype=bool)
        if not (len(output) == len(geometric_score) == len(geometric_eligible)):
            raise ValueError("point and geometric arrays must align")
    for cluster_id in np.unique(clusters[clusters >= 0]):
        member = clusters == cluster_id
        proposal = top_fraction(output[member], fraction)
        if geometric_score is not None:
            coverage = float(geometric_eligible[member].mean())
            geometry = top_fraction(geometric_score[member], fraction)
            if coverage >= minimum_coverage and geometry >= geometric_floor:
                proposal = min(1.0, proposal + float(boost) * geometry)
        output[member] = np.maximum(output[member], proposal)
    return output


def best_result(tp: np.ndarray, fp: np.ndarray, total_positive: int,
                thresholds: np.ndarray):
    fn = total_positive - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    index = int(np.argmax(iou))
    tpi, fpi, fni = int(tp[index]), int(fp[index]), int(fn[index])
    return {
        "iou": float(iou[index]),
        "threshold": float(thresholds[index]),
        "precision": tpi / max(tpi + fpi, 1),
        "recall": tpi / max(tpi + fni, 1),
        "tp": tpi, "fp": fpi, "fn": fni,
    }


def clipped_reference_scan(dataset: SemanticKITTI4D, sequence: int,
                           frame: int) -> np.ndarray:
    scan_path, _ = dataset._frame_paths(sequence, frame)
    xyz = _read_scan(scan_path)[:, :3]
    if dataset.point_range is not None:
        keep = np.all(np.abs(xyz) < dataset.point_range, axis=1)
        xyz = xyz[keep]
    return xyz.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument("--history-offsets", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12, 16, 20])
    parser.add_argument("--range-deltas", type=float, nargs="+",
                        default=[0.30, 0.50, 0.75])
    parser.add_argument("--minimum-history-hits", type=int, nargs="+",
                        default=[2, 4])
    parser.add_argument("--boosts", type=float, nargs="+",
                        default=[0.10, 0.25, 0.50])
    parser.add_argument("--geometry-floors", type=float, nargs="+",
                        default=[0.50, 0.75])
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--minimum-coverage", type=float, default=0.25)
    parser.add_argument("--precision-drop", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    d, p, model_cfg = cfg["dataset"], cfg["pose"], cfg["model"]
    if len(d["val_sequences"]) != 1:
        raise ValueError("causal evaluation requires exactly one sequence")
    if not model_cfg.get("use_cluster", False):
        raise ValueError("geometric proposals require the cluster head")
    offsets = sorted(set(int(x) for x in args.history_offsets))
    if not offsets or offsets[0] < 1:
        raise ValueError("history offsets must be positive")
    thresholds = np.asarray(sorted(set(args.thresholds)), np.float32)
    candidates = [
        (float(delta), int(hits), float(boost), float(floor))
        for delta in args.range_deltas
        for hits in args.minimum_history_hits
        for boost in args.boosts
        for floor in args.geometry_floors
    ]

    dataset = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], p.get("mode", "gt"),
        p.get("rot_std_deg", 0.0), p.get("trans_std_m", 0.0),
        p.get("seed", 0), d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"),
        feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        return_point_map=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=args.num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, d, model_cfg, device)

    # B0 point, B1 Top25, followed by geometric candidates.
    tp = np.zeros((2 + len(candidates), len(thresholds)), np.int64)
    fp = np.zeros_like(tp)
    total_positive = 0
    scan_cache: dict[int, np.ndarray] = {}
    cache_order: deque[int] = deque()
    previous_meta = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_frames is not None and batch_index >= args.max_frames:
                break
            sequence, frame = (int(value) for value in batch["meta"][0])
            if previous_meta is not None and (
                    sequence != previous_meta[0] or frame != previous_meta[1] + 1):
                raise RuntimeError(
                    f"non-causal order: {previous_meta} -> {(sequence, frame)}")
            previous_meta = (sequence, frame)

            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], 1)[:, 1].cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            point_voxel = batch["ref_point_voxel"].numpy()
            point_score_all = voxel_score[point_voxel]
            point_cluster_all = cluster_ids[point_voxel]

            current_xyz = clipped_reference_scan(dataset, sequence, frame)
            if len(current_xyz) != len(point_voxel):
                raise RuntimeError(
                    f"scan/point-map mismatch at frame {frame}: "
                    f"{len(current_xyz)} != {len(point_voxel)}")
            provider = dataset.pose_providers[sequence]
            past_current = []
            for offset in offsets:
                past_frame = frame - offset
                if past_frame in scan_cache:
                    transformed = _transform(
                        scan_cache[past_frame],
                        provider.relative(past_frame, frame)).astype(np.float32)
                    if dataset.point_range is not None:
                        keep = np.all(
                            np.abs(transformed) < dataset.point_range, axis=1)
                        transformed = transformed[keep]
                    past_current.append(transformed)
            gaps = causal_range_gaps(
                current_xyz, past_current, args.neighborhood_radius)

            gt = batch["ref_point_motion"].numpy()
            valid = gt != -1
            positive = gt[valid] == 1
            total_positive += int(positive.sum())
            point_score = point_score_all[valid]
            point_cluster = point_cluster_all[valid]
            scores = [
                point_score,
                complete_cluster_scores(point_score, point_cluster),
            ]
            for delta, hits, boost, floor in candidates:
                geometry_all, eligible_all = geometric_disagreement(
                    gaps, delta, hits)
                scores.append(complete_cluster_scores(
                    point_score, point_cluster,
                    geometry_all[valid], eligible_all[valid],
                    boost=boost, geometric_floor=floor,
                    minimum_coverage=args.minimum_coverage))
            for row, score in enumerate(scores):
                batch_tp, batch_fp = threshold_counts(
                    score, positive, thresholds)
                tp[row] += batch_tp
                fp[row] += batch_fp

            scan_cache[frame] = current_xyz
            cache_order.append(frame)
            while cache_order and cache_order[0] < frame - offsets[-1]:
                del scan_cache[cache_order.popleft()]
            if batch_index % 100 == 0:
                print(f"  frame {batch_index}/{len(loader)}", flush=True)

    labels = [("point", None), ("top25", None)] + [
        ("geometry", candidate) for candidate in candidates]
    rows = []
    for row, (mode, candidate) in enumerate(labels):
        result = best_result(tp[row], fp[row], total_positive, thresholds)
        rows.append((mode, candidate, result))
    baseline, top25 = rows[0][2], rows[1][2]
    precision_floor = top25["precision"] - float(args.precision_drop)
    acceptable = [row for row in rows[2:]
                  if row[2]["precision"] >= precision_floor]
    best = max(acceptable, key=lambda row: row[2]["iou"]) if acceptable else None
    unconstrained = max(rows[2:], key=lambda row: row[2]["iou"])

    print(f"\n=== native causal geometric proposals, val seq "
          f"{d['val_sequences']} ===")
    print("Prediction uses frozen network + raw scans + odometry only; "
          "GT is metric-only.")
    print(f"history_offsets={offsets} neighborhood={args.neighborhood_radius} "
          f"coverage>={args.minimum_coverage:.2f} "
          f"precision_drop<={args.precision_drop:.3f}")
    print(f"B0 point: IoU={baseline['iou']:.4f} "
          f"P={baseline['precision']:.4f} R={baseline['recall']:.4f} "
          f"@ {baseline['threshold']:.5f}")
    print(f"B1 Top25: IoU={top25['iou']:.4f} "
          f"P={top25['precision']:.4f} R={top25['recall']:.4f} "
          f"@ {top25['threshold']:.5f}")
    print(f"{'delta':>7} {'hits':>5} {'boost':>7} {'floor':>7} "
          f"{'IoU':>8} {'@th':>9} {'Prec':>8} {'Rec':>8} {'ok':>4}")
    for _, candidate, result in sorted(
            rows[2:], key=lambda row: row[2]["iou"], reverse=True):
        delta, hits, boost, floor = candidate
        ok = result["precision"] >= precision_floor
        print(f"{delta:7.2f} {hits:5d} {boost:7.2f} {floor:7.2f} "
              f"{result['iou']:8.4f} {result['threshold']:9.5f} "
              f"{result['precision']:8.4f} {result['recall']:8.4f} "
              f"{('yes' if ok else 'no'):>4}")
    candidate, result = unconstrained[1], unconstrained[2]
    print("\nBEST UNCONSTRAINED: "
          f"delta={candidate[0]:.2f} hits={candidate[1]} "
          f"boost={candidate[2]:.2f} floor={candidate[3]:.2f} "
          f"IoU={result['iou']:.4f} P={result['precision']:.4f} "
          f"R={result['recall']:.4f}")
    if best is None:
        print("DECISION: REJECT -- no candidate satisfies precision control")
    else:
        candidate, result = best[1], best[2]
        delta = result["iou"] - top25["iou"]
        decision = "ACCEPT" if delta > 0 else "REJECT"
        print(f"DECISION: {decision} precision-controlled geometry")
        print(f"  delta={candidate[0]:.2f} hits={candidate[1]} "
              f"boost={candidate[2]:.2f} floor={candidate[3]:.2f}")
        print(f"  IoU={result['iou']:.4f} delta_over_B1={delta:+.4f} "
              f"threshold={result['threshold']:.5f}")
        print(f"  P={result['precision']:.4f} R={result['recall']:.4f}")


if __name__ == "__main__":
    main()
