#!/usr/bin/env python3
"""Audit a fixed 3x3 occlusion-aware neighborhood residual representation.

Past scans are ego-registered exactly as in the frozen dataset.  For each
current point, instead of comparing only the identical range-image pixel, this
audit searches the fixed 3x3 angular neighborhood and selects the valid past
return with minimum absolute range discrepancy.  The selected discrepancy
keeps its sign.  Empty neighborhoods produce residual zero plus invalidity.

The alternative representation is reduced to one representative per occupied
voxel using the same label-free maximum-residual rule as the frozen dataset.
Predicted semantic clusters and the frozen checkpoint remain unchanged.
Ground truth supplies diagnostic majority-motion cluster labels only.

No radius, threshold or model parameter is selected.  A representation probe
passes only if one predefined neighborhood-residual statistic separates hard
moving clusters from static clusters with AUROC >= 0.70 on every development
sequence.  Audited sequence 08 is forbidden.
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import OrderedDict, defaultdict
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from experiments.multisequence_cluster_residual_audit import (
    auroc,
    cluster_residual_features,
    percentile_text,
)
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.residual_features import spherical_project
from sparseunet4d.datasets.semantickitti import _read_scan, _transform


FEATURES = (
    "nrr_abs_top25",
    "nrr_abs_top25_normalized",
    "nrr_coherent",
    "nrr_coherent_normalized",
    "nrr_sign_consistency",
    "nrr_offset_support",
    "nrr_consistency",
    "nrr_valid_fraction",
    "nrr_consistency_validated",
)


class ScanCache:
    """Small per-process LRU cache; avoids repeatedly reading overlapping scans."""

    def __init__(self, maximum_items: int = 24):
        self.maximum_items = int(maximum_items)
        self.items: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get(self, root: str, sequence: int, frame: int) -> np.ndarray:
        key = (int(sequence), int(frame))
        if key in self.items:
            value = self.items.pop(key)
            self.items[key] = value
            return value
        path = os.path.join(
            root, f"{sequence:02d}", "velodyne", f"{frame:06d}.bin"
        )
        value = _read_scan(path)
        self.items[key] = value
        while len(self.items) > self.maximum_items:
            self.items.popitem(last=False)
        return value


def neighborhood_residual(
    query_xyz: np.ndarray,
    past_xyz: np.ndarray,
    radius_pixels: int = 1,
    height: int = 64,
    width: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed minimum-|range difference| and match validity per query."""
    query_xyz = np.asarray(query_xyz, np.float32)
    past_xyz = np.asarray(past_xyz, np.float32)
    result = np.zeros(len(query_xyz), np.float32)
    matched = np.zeros(len(query_xyz), bool)
    if len(query_xyz) == 0 or len(past_xyz) == 0:
        return result, matched

    _, query_u, query_v, query_range, query_valid = spherical_project(
        query_xyz, H=height, W=width
    )
    past_image, _, _, _, _ = spherical_project(
        past_xyz, H=height, W=width
    )
    best_absolute = np.full(len(query_xyz), np.inf, np.float64)
    best_signed = np.zeros(len(query_xyz), np.float64)

    for delta_v in range(-radius_pixels, radius_pixels + 1):
        candidate_v = query_v + delta_v
        vertical_valid = (candidate_v >= 0) & (candidate_v < height)
        safe_v = np.clip(candidate_v, 0, height - 1)
        for delta_u in range(-radius_pixels, radius_pixels + 1):
            # Horizontal image coordinate is circular in yaw.
            candidate_u = (query_u + delta_u) % width
            past_range = past_image[safe_v, candidate_u]
            valid = (
                query_valid & vertical_valid & np.isfinite(past_range)
            )
            signed = query_range - past_range
            absolute = np.abs(signed)
            better = valid & (absolute < best_absolute)
            best_absolute[better] = absolute[better]
            best_signed[better] = signed[better]

    matched = np.isfinite(best_absolute)
    result[matched] = best_signed[matched].astype(np.float32)
    return result, matched


def representative_rows(
    point_voxel: np.ndarray,
    point_residual: np.ndarray,
) -> np.ndarray:
    """Choose the maximum-|residual| point per voxel, matching dataset policy."""
    if len(point_voxel) == 0:
        return np.zeros(0, np.int64)
    strength = np.abs(point_residual).max(axis=1)
    order = np.argsort(strength, kind="stable")
    representative: dict[int, int] = {}
    for row in order.tolist():
        representative[int(point_voxel[row])] = int(row)
    return np.asarray(list(representative.values()), np.int64)


@dataclasses.dataclass(frozen=True)
class Record:
    sequence: int
    moving: bool
    hard_moving: bool
    values: dict[str, float]
    saturation_fraction: float


def group_auc(records: list[Record], feature: str) -> float:
    negative = [record for record in records if not record.moving]
    positive = [record for record in records if record.hard_moving]
    selected = negative + positive
    labels = np.asarray([record.moving for record in selected], bool)
    scores = np.asarray([record.values[feature] for record in selected])
    return auroc(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hard-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-hard-auroc", type=float, default=0.70)
    parser.add_argument("--signal-floor-m", type=float, default=0.10)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if abs(args.hard_threshold - 0.20) > 1e-12:
        raise ValueError("frozen B1 hard threshold must remain 0.20")
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    dataset_cfg = cfg["dataset"]
    pose_cfg = cfg["pose"]
    model_cfg = cfg["model"]
    sequences = sorted(int(x) for x in dataset_cfg["val_sequences"])
    if len(sequences) < 2:
        raise ValueError("at least two development sequences are required")
    if 8 in sequences:
        raise ValueError("audited sequence 08 is forbidden for diagnostics")
    if dataset_cfg.get("feat_rep") != "residual":
        raise ValueError("audit expects the frozen label-free residual setup")
    offsets = list(
        dataset_cfg.get(
            "frame_offsets", range(1, dataset_cfg["n_frames"])
        )
    )
    if len(offsets) != dataset_cfg["n_frames"] - 1:
        raise ValueError("frame offset count must equal n_frames - 1")

    dataset = SemanticKITTI4D(
        dataset_cfg["root"], sequences, dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"], dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg["seed"], dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        frame_offsets=offsets,
        feat_rep=dataset_cfg.get("feat_rep", "label"),
        residual_validity=dataset_cfg.get("residual_validity", False),
        residual_all_frames=dataset_cfg.get("residual_all_frames", False),
        return_point_map=True,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=me_collate,
        num_workers=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, dataset_cfg, model_cfg, device)
    cache = ScanCache()
    records: list[Record] = []
    ignored = defaultdict(int)
    clip_value = float(dataset_cfg.get("res_clip", 3.0))
    point_range = float(dataset_cfg["point_range"])

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            output = model(to_st(batch, device))
            voxel_probability = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            cluster_motion = pooled_scores(
                voxel_probability, cluster_ids, "top25"
            )
            point_voxel = batch["ref_point_voxel"].numpy()
            point_motion = batch["ref_point_motion"].numpy()
            point_cluster = cluster_ids[point_voxel]

            current_scan = cache.get(
                dataset_cfg["root"], sequence, frame
            )
            current_xyz = current_scan[:, :3]
            current_keep = np.all(np.abs(current_xyz) < point_range, axis=1)
            current_xyz = current_xyz[current_keep]
            if len(current_xyz) != len(point_voxel):
                raise RuntimeError(
                    f"point alignment failed at {sequence:02d}/{frame:06d}"
                )

            query_mask = point_cluster >= 0
            query_xyz = current_xyz[query_mask]
            query_voxel = point_voxel[query_mask]
            query_cluster = point_cluster[query_mask]
            query_motion = point_motion[query_mask]
            residual = np.zeros((len(query_xyz), len(offsets)), np.float32)
            validity = np.zeros_like(residual, bool)
            provider = dataset.pose_providers[sequence]
            for offset_index, offset in enumerate(offsets):
                past_frame = frame - int(offset)
                if past_frame < 0:
                    continue
                past_scan = cache.get(
                    dataset_cfg["root"], sequence, past_frame
                )
                transform = provider.relative(past_frame, frame)
                past_xyz = _transform(past_scan[:, :3], transform).astype(
                    np.float32
                )
                past_keep = np.all(np.abs(past_xyz) < point_range, axis=1)
                values, valid = neighborhood_residual(
                    query_xyz, past_xyz[past_keep], radius_pixels=1
                )
                residual[:, offset_index] = np.clip(
                    values, -clip_value, clip_value
                )
                validity[:, offset_index] = valid

            for cluster_id in np.unique(query_cluster):
                cluster_id = int(cluster_id)
                point_rows = np.flatnonzero(query_cluster == cluster_id)
                valid_gt = point_rows[query_motion[point_rows] != -1]
                if len(valid_gt) == 0:
                    ignored[sequence] += 1
                    continue
                moving_fraction = float(
                    (query_motion[valid_gt] == 1).mean()
                )
                representatives = representative_rows(
                    query_voxel[point_rows], residual[point_rows]
                )
                cluster_residual = residual[point_rows][representatives]
                cluster_validity = validity[point_rows][representatives]
                cluster_xyz = query_xyz[point_rows][representatives]
                median_range = float(
                    np.median(np.linalg.norm(cluster_xyz, axis=1))
                )
                raw_features, saturation = cluster_residual_features(
                    cluster_residual, median_range, args.signal_floor_m,
                    clip_value,
                )
                valid_fraction = float(cluster_validity.mean())
                values = {
                    "nrr_abs_top25": raw_features["residual_abs_top25"],
                    "nrr_abs_top25_normalized": raw_features[
                        "residual_abs_top25_normalized"
                    ],
                    "nrr_coherent": raw_features["residual_coherent"],
                    "nrr_coherent_normalized": raw_features[
                        "residual_coherent_normalized"
                    ],
                    "nrr_sign_consistency": raw_features[
                        "residual_sign_consistency"
                    ],
                    "nrr_offset_support": raw_features[
                        "residual_offset_support"
                    ],
                    "nrr_consistency": raw_features[
                        "residual_consistency"
                    ],
                    "nrr_valid_fraction": valid_fraction,
                    "nrr_consistency_validated": (
                        raw_features["residual_consistency"] * valid_fraction
                    ),
                }
                motion_score = float(cluster_motion[cluster_id])
                moving = moving_fraction >= 0.50
                records.append(
                    Record(
                        sequence=sequence,
                        moving=moving,
                        hard_moving=moving and motion_score < args.hard_threshold,
                        values=values,
                        saturation_fraction=saturation,
                    )
                )

            if frame_index % 100 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)
            if args.max_frames and frame_index + 1 >= args.max_frames:
                break

    print("\n=== fixed 3x3 neighborhood-residual separability audit ===")
    print(
        f"sequences={sequences} offsets={offsets} radius=1 "
        f"hard_motion_top25<{args.hard_threshold:.2f} "
        f"required_worst_AUROC={args.minimum_hard_auroc:.2f}"
    )
    print(
        "Frozen clusters/checkpoint; GT is diagnostic-only; no radius or "
        "decision threshold search."
    )

    print("\n---- populations ----")
    print(f"{'sequence':>10} {'static':>10} {'moving':>10} {'hard moving':>12}")
    for sequence in sequences:
        selected = [record for record in records if record.sequence == sequence]
        print(
            f"{sequence:10d} {sum(not r.moving for r in selected):10d} "
            f"{sum(r.moving for r in selected):10d} "
            f"{sum(r.hard_moving for r in selected):12d}"
        )

    print("\n---- HARD-moving versus static cluster AUROC ----")
    sequence_header = " ".join(f"seq{sequence:02d}" for sequence in sequences)
    print(f"{'feature':>36} {'worst':>8} {sequence_header}")
    feature_auc = {}
    for feature in FEATURES:
        values = []
        for sequence in sequences:
            selected = [
                record for record in records if record.sequence == sequence
            ]
            values.append(group_auc(selected, feature))
        feature_auc[feature] = values
        finite = [value for value in values if np.isfinite(value)]
        worst = min(finite) if len(finite) == len(values) else float("nan")
        print(
            f"{feature:>36} {worst:8.4f} "
            + " ".join(f"{value:7.4f}" for value in values)
        )

    print("\n---- distributions ----")
    for sequence in sequences:
        selected = [record for record in records if record.sequence == sequence]
        groups = {
            "static": [record for record in selected if not record.moving],
            "hard-moving": [record for record in selected if record.hard_moving],
            "easy-moving": [
                record for record in selected
                if record.moving and not record.hard_moving
            ],
        }
        print(f"\nsequence {sequence:02d}")
        for feature in FEATURES:
            print(f"  {feature}")
            for name, group in groups.items():
                print(
                    f"    {name:>11}: "
                    f"{percentile_text([r.values[feature] for r in group])}"
                )
        print("  saturation fraction")
        for name, group in groups.items():
            print(
                f"    {name:>11}: "
                f"{percentile_text([r.saturation_fraction for r in group])}"
            )

    candidates = []
    for feature, values in feature_auc.items():
        if all(np.isfinite(value) for value in values):
            candidates.append((min(values), float(np.mean(values)), feature))
    winner = max(candidates, default=(float("nan"), float("nan"), "none"))
    print("\n=== REPRESENTATION DECISION ===")
    if np.isfinite(winner[0]) and winner[0] >= args.minimum_hard_auroc:
        print(
            "ACCEPT fixed 3x3 neighborhood-residual training probe\n"
            f"best_feature={winner[2]} worst_hard_AUROC={winner[0]:.4f} "
            f"macro_hard_AUROC={winner[1]:.4f}"
        )
    else:
        print(
            "REJECT fixed 3x3 neighborhood residual\n"
            f"best_feature={winner[2]} worst_hard_AUROC={winner[0]:.4f}\n"
            "Do not retrain with this representation."
        )


if __name__ == "__main__":
    main()
