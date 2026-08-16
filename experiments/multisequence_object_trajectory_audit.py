#!/usr/bin/env python3
"""Frozen object-trajectory separability audit across temporal LiDAR slices.

Predicted movable-semantic voxels are clustered independently in every input
time slice. Because past scans are already ego-registered into the reference
frame, a static object's cluster centroid should remain stable while a moving
object traces a displacement trajectory. Reference clusters are associated to
same-class past clusters with one-to-one Hungarian matching, a physical
speed-based distance gate and a size-consistency gate.

The checkpoint, semantic predictions, reference clusters and B1 threshold are
frozen. Ground truth supplies diagnostic majority-motion labels only. No model
parameter or decision threshold is selected. A trajectory branch is accepted
only if the predeclared composite ``trajectory_score`` separates hard moving
clusters from static clusters with AUROC >= 0.70 on every development
sequence. The component statistics are diagnostic only. Sequence 08 is
explicitly forbidden.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from experiments.multisequence_cluster_residual_audit import auroc, percentile_text
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.cluster_head import connected_components_vox


MOVABLE_LEARNING_IDS = np.arange(1, 9, dtype=np.int64)
FEATURES = (
    "maximum_displacement",
    "furthest_displacement",
    "fitted_speed",
    "median_speed",
    "direction_consistency",
    "speed_consistency",
    "monotonicity",
    "track_support",
    "displacement_support",
    "trajectory_score",
)


@dataclasses.dataclass(frozen=True)
class ObjectDescriptor:
    cluster_id: int
    centroid: np.ndarray
    radius: float
    semantic_class: int
    voxel_count: int


@dataclasses.dataclass(frozen=True)
class MatchObservation:
    offset_scans: int
    displacement: np.ndarray
    size_ratio: float


@dataclasses.dataclass(frozen=True)
class Record:
    sequence: int
    moving: bool
    hard_moving: bool
    values: dict[str, float]


def descriptors_from_cluster_ids(
    coordinates: np.ndarray,
    semantic_prediction: np.ndarray,
    cluster_ids: np.ndarray,
    voxel_size: float,
) -> list[ObjectDescriptor]:
    descriptors = []
    for cluster_id in np.unique(cluster_ids[cluster_ids >= 0]):
        cluster_id = int(cluster_id)
        rows = np.flatnonzero(cluster_ids == cluster_id)
        xyz = (
            coordinates[rows, 1:4].astype(np.float32) + 0.5
        ) * voxel_size
        centroid = xyz.mean(axis=0)
        radial = np.linalg.norm(xyz - centroid, axis=1)
        classes, counts = np.unique(
            semantic_prediction[rows], return_counts=True
        )
        descriptors.append(
            ObjectDescriptor(
                cluster_id=cluster_id,
                centroid=centroid,
                radius=float(max(np.percentile(radial, 90), voxel_size)),
                semantic_class=int(classes[int(np.argmax(counts))]),
                voxel_count=len(rows),
            )
        )
    return descriptors


def cluster_time_slice(
    coordinates: np.ndarray,
    semantic_prediction: np.ndarray,
    time_index: int,
    voxel_size: float,
    link: int,
    minimum_size: int,
) -> list[ObjectDescriptor]:
    selected = (
        (coordinates[:, 4] == time_index)
        & np.isin(semantic_prediction, MOVABLE_LEARNING_IDS)
    )
    rows = np.flatnonzero(selected)
    if len(rows) == 0:
        return []
    local_labels = connected_components_vox(
        coordinates[rows, 1:4], link=link, min_size=minimum_size
    )
    global_labels = np.full(len(coordinates), -1, np.int64)
    valid = local_labels >= 0
    global_labels[rows[valid]] = local_labels[valid]
    return descriptors_from_cluster_ids(
        coordinates, semantic_prediction, global_labels, voxel_size
    )


def associate_objects(
    reference: list[ObjectDescriptor],
    past: list[ObjectDescriptor],
    offset_scans: int,
    scan_period_seconds: float,
    maximum_speed_mps: float,
    base_gate_m: float,
    maximum_size_ratio: float,
) -> dict[int, MatchObservation]:
    if not reference or not past:
        return {}
    maximum_distance = max(
        base_gate_m,
        maximum_speed_mps * scan_period_seconds * offset_scans,
    )
    large = 1e9
    cost = np.full((len(reference), len(past)), large, np.float64)
    displacement = np.zeros((len(reference), len(past), 3), np.float32)
    size_ratio = np.full((len(reference), len(past)), np.inf, np.float64)
    for reference_index, current in enumerate(reference):
        for past_index, previous in enumerate(past):
            vector = current.centroid - previous.centroid
            distance = float(np.linalg.norm(vector))
            ratio = max(
                (current.radius + 1e-3) / (previous.radius + 1e-3),
                (previous.radius + 1e-3) / (current.radius + 1e-3),
            )
            displacement[reference_index, past_index] = vector
            size_ratio[reference_index, past_index] = ratio
            if (
                current.semantic_class == previous.semantic_class
                and ratio <= maximum_size_ratio
                and distance <= maximum_distance
            ):
                cost[reference_index, past_index] = (
                    distance + 0.25 * abs(np.log(ratio))
                )
    reference_rows, past_columns = linear_sum_assignment(cost)
    matches = {}
    for reference_index, past_index in zip(
        reference_rows.tolist(), past_columns.tolist()
    ):
        if cost[reference_index, past_index] >= large:
            continue
        cluster_id = reference[reference_index].cluster_id
        matches[cluster_id] = MatchObservation(
            offset_scans=int(offset_scans),
            displacement=displacement[reference_index, past_index].copy(),
            size_ratio=float(size_ratio[reference_index, past_index]),
        )
    return matches


def trajectory_features(
    observations: list[MatchObservation],
    available_offset_count: int,
    scan_period_seconds: float,
) -> dict[str, float]:
    if not observations:
        return {feature: 0.0 for feature in FEATURES}
    observations = sorted(observations, key=lambda item: item.offset_scans)
    offsets = np.asarray(
        [item.offset_scans for item in observations], np.float64
    )
    times = offsets * scan_period_seconds
    vectors = np.stack([item.displacement for item in observations]).astype(
        np.float64
    )
    distances = np.linalg.norm(vectors, axis=1)
    speeds = distances / np.maximum(times, 1e-6)
    maximum_displacement = float(distances.max())
    furthest_displacement = float(distances[-1])
    fitted_speed = float(
        np.dot(times, distances) / max(np.dot(times, times), 1e-9)
    )
    median_speed = float(np.median(speeds))
    nonzero = distances > 1e-4
    if nonzero.any():
        unit = vectors[nonzero] / distances[nonzero, None]
        direction_consistency = float(np.linalg.norm(unit.mean(axis=0)))
    else:
        direction_consistency = 0.0
    speed_mean = float(speeds.mean())
    speed_consistency = float(
        1.0 / (1.0 + speeds.std() / max(speed_mean, 1e-3))
    )
    if len(distances) <= 1:
        monotonicity = 1.0
    else:
        monotonicity = float((np.diff(distances) >= -0.10).mean())
    track_support = len(observations) / max(available_offset_count, 1)
    size_consistency = float(
        np.exp(-np.mean([abs(np.log(item.size_ratio)) for item in observations]))
    )
    displacement_support = maximum_displacement * track_support
    trajectory_score = (
        fitted_speed
        * direction_consistency
        * speed_consistency
        * monotonicity
        * track_support
        * size_consistency
    )
    return {
        "maximum_displacement": maximum_displacement,
        "furthest_displacement": furthest_displacement,
        "fitted_speed": fitted_speed,
        "median_speed": median_speed,
        "direction_consistency": direction_consistency,
        "speed_consistency": speed_consistency,
        "monotonicity": monotonicity,
        "track_support": track_support,
        "displacement_support": displacement_support,
        "trajectory_score": float(trajectory_score),
    }


def hard_auc(records: list[Record], feature: str) -> float:
    selected = [
        record for record in records
        if not record.moving or record.hard_moving
    ]
    labels = np.asarray([record.moving for record in selected], bool)
    scores = np.asarray([record.values[feature] for record in selected])
    return auroc(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hard-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-hard-auroc", type=float, default=0.70)
    parser.add_argument("--scan-period-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-speed-mps", type=float, default=20.0)
    parser.add_argument("--base-gate-m", type=float, default=2.0)
    parser.add_argument("--maximum-size-ratio", type=float, default=2.0)
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
    offsets = list(
        dataset_cfg.get(
            "frame_offsets", range(1, dataset_cfg["n_frames"])
        )
    )
    model_link = int(model_cfg.get("cluster_link", 2))
    model_minimum_size = int(model_cfg.get("cluster_min_size", 3))

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
    records: list[Record] = []
    population = {
        sequence: {"static": 0, "moving": 0, "hard": 0, "unmatched": 0}
        for sequence in sequences
    }

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            output = model(to_st(batch, device))
            coordinates = output["coords"].detach().cpu().numpy()
            semantic_prediction = output["semantic_logits"].argmax(1).cpu().numpy()
            voxel_probability = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            reference_cluster_ids = output["cluster_row_id"].cpu().numpy()
            cluster_motion = pooled_scores(
                voxel_probability, reference_cluster_ids, "top25"
            )
            reference = descriptors_from_cluster_ids(
                coordinates, semantic_prediction, reference_cluster_ids,
                float(dataset_cfg["voxel_size"]),
            )
            reference_by_id = {
                descriptor.cluster_id: descriptor for descriptor in reference
            }
            track_observations = {
                descriptor.cluster_id: [] for descriptor in reference
            }
            available_offsets = []
            for time_index, offset in enumerate(offsets, start=1):
                if not np.any(coordinates[:, 4] == time_index):
                    continue
                available_offsets.append(offset)
                past = cluster_time_slice(
                    coordinates, semantic_prediction, time_index,
                    float(dataset_cfg["voxel_size"]), model_link,
                    model_minimum_size,
                )
                matches = associate_objects(
                    reference, past, offset, args.scan_period_seconds,
                    args.maximum_speed_mps, args.base_gate_m,
                    args.maximum_size_ratio,
                )
                for cluster_id, observation in matches.items():
                    track_observations[cluster_id].append(observation)

            point_voxel = batch["ref_point_voxel"].numpy()
            point_motion = batch["ref_point_motion"].numpy()
            point_cluster = reference_cluster_ids[point_voxel]
            for cluster_id, descriptor in reference_by_id.items():
                point_rows = np.flatnonzero(point_cluster == cluster_id)
                valid = point_rows[point_motion[point_rows] != -1]
                if len(valid) == 0:
                    continue
                moving_fraction = float((point_motion[valid] == 1).mean())
                moving = moving_fraction >= 0.50
                motion_score = float(cluster_motion[cluster_id])
                hard = moving and motion_score < args.hard_threshold
                observations = track_observations[cluster_id]
                if not observations:
                    population[sequence]["unmatched"] += 1
                values = trajectory_features(
                    observations, len(available_offsets),
                    args.scan_period_seconds,
                )
                records.append(Record(sequence, moving, hard, values))
                population[sequence]["moving" if moving else "static"] += 1
                if hard:
                    population[sequence]["hard"] += 1

            if frame_index % 100 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)
            if args.max_frames and frame_index + 1 >= args.max_frames:
                break

    print("\n=== frozen object-trajectory separability audit ===")
    print(
        f"sequences={sequences} offsets={offsets} "
        f"scan_period={args.scan_period_seconds:.2f}s "
        f"max_speed={args.maximum_speed_mps:.1f}m/s "
        f"size_ratio<={args.maximum_size_ratio:.1f} "
        f"required_worst_AUROC={args.minimum_hard_auroc:.2f}"
    )
    print(
        "Frozen network/clusters; GT is diagnostic-only; no association or "
        "decision parameter search."
    )
    print("\n---- populations ----")
    print(
        f"{'sequence':>10} {'static':>10} {'moving':>10} {'hard':>10} "
        f"{'unmatched':>10}"
    )
    for sequence in sequences:
        item = population[sequence]
        print(
            f"{sequence:10d} {item['static']:10d} {item['moving']:10d} "
            f"{item['hard']:10d} {item['unmatched']:10d}"
        )

    if args.max_frames:
        print(
            "\n=== SMOKE CHECK COMPLETE ===\n"
            f"Processed the requested {args.max_frames} frames without a "
            "runtime error.\n"
            "No AUROC or architecture decision is computed from a truncated "
            "dataset. Run again without --max-frames for the frozen gate."
        )
        return

    print("\n---- HARD-moving versus static trajectory AUROC ----")
    sequence_header = " ".join(f"seq{sequence:02d}" for sequence in sequences)
    print(f"{'feature':>30} {'worst':>8} {sequence_header}")
    feature_auc = {}
    for feature in FEATURES:
        values = []
        for sequence in sequences:
            selected = [record for record in records if record.sequence == sequence]
            values.append(hard_auc(selected, feature))
        feature_auc[feature] = values
        finite = [value for value in values if np.isfinite(value)]
        worst = min(finite) if len(finite) == len(values) else float("nan")
        print(
            f"{feature:>30} {worst:8.4f} "
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

    diagnostic_candidates = []
    for feature, values in feature_auc.items():
        if all(np.isfinite(value) for value in values):
            diagnostic_candidates.append(
                (min(values), float(np.mean(values)), feature)
            )
    diagnostic_winner = max(
        diagnostic_candidates,
        default=(float("nan"), float("nan"), "none"),
    )
    primary_values = feature_auc["trajectory_score"]
    primary_finite = all(np.isfinite(value) for value in primary_values)
    primary_worst = (
        min(primary_values) if primary_finite else float("nan")
    )
    primary_macro = (
        float(np.mean(primary_values)) if primary_finite else float("nan")
    )
    print("\n=== FINAL ARCHITECTURE GATE ===")
    print(
        "predeclared_feature=trajectory_score "
        f"worst_hard_AUROC={primary_worst:.4f} "
        f"macro_hard_AUROC={primary_macro:.4f}"
    )
    print(
        "best_diagnostic_feature="
        f"{diagnostic_winner[2]} "
        f"worst_hard_AUROC={diagnostic_winner[0]:.4f}"
    )
    if (
        np.isfinite(primary_worst)
        and primary_worst >= args.minimum_hard_auroc
    ):
        print(
            "ACCEPT object-trajectory branch\n"
            "The predeclared trajectory_score passed the robust gate."
        )
    else:
        print(
            "REJECT object-trajectory branch\n"
            "The predeclared trajectory_score did not pass the robust gate.\n"
            "Freeze B1 Top25 as the proposed method and move to paper tables, "
            "external evaluation and writing."
        )


if __name__ == "__main__":
    main()
