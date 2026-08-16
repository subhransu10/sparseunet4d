#!/usr/bin/env python3
"""Observability audit for B1-missed moving instance-frames.

The B1 method is frozen as Top25 cluster completion at threshold 0.20.  This
script performs no model selection and no parameter sweep.  Ground truth is
used only to identify and describe missed moving instance-frames.

For every moving instance-frame with at least five points, the audit records:

* B1 detected fraction;
* coverage by predicted movable semantics and by a valid predicted cluster;
* point-motion probability summaries;
* signed temporal-residual strength summaries;
* range, raw mover class and point count.

Misses are split into three mutually exclusive, actionable causes:

1. semantic foreground absent;
2. semantic foreground present, but no valid cluster survives;
3. a cluster exists, but its completed motion score remains too low.

Audited sequence 08 is explicitly forbidden.
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import split_label
from sparseunet4d.datasets.semantickitti import _read_label, _read_scan


MOVING_NAMES = {
    252: "car",
    253: "bicyclist",
    254: "person",
    255: "motorcyclist",
    256: "on-rails",
    257: "bus",
    258: "truck",
    259: "other-vehicle",
}
MOVABLE_LEARNING_IDS = np.arange(1, 9, dtype=np.int64)
RANGE_BINS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 52))


def top_fraction(values: np.ndarray, fraction: float = 0.25) -> float:
    if len(values) == 0:
        return 0.0
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, len(values) - count)[-count:].mean())


def complete_top25(
    point_score: np.ndarray,
    point_cluster: np.ndarray,
    cluster_score: np.ndarray,
) -> np.ndarray:
    completed = point_score.copy()
    member = (point_cluster >= 0) & (point_cluster < len(cluster_score))
    completed[member] = np.maximum(
        completed[member], cluster_score[point_cluster[member]]
    )
    return completed


@dataclasses.dataclass(frozen=True)
class Observation:
    sequence: int
    frame: int
    raw_class: int
    instance_id: int
    point_count: int
    median_range: float
    b1_fraction: float
    semantic_coverage: float
    cluster_coverage: float
    probability_mean: float
    probability_top25: float
    probability_maximum: float
    residual_mean: float
    residual_top25: float
    residual_maximum: float

    @property
    def group(self) -> str:
        if self.b1_fraction < 0.10:
            return "missed"
        if self.b1_fraction < 0.90:
            return "partial"
        return "full"

    @property
    def failure_cause(self) -> str:
        if self.group != "missed":
            return "not-missed"
        if self.semantic_coverage < 0.10:
            return "semantic absent"
        if self.cluster_coverage < 0.10:
            return "cluster absent"
        return "cluster low-score"


def percentile_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    array = np.asarray(values, np.float64)
    q10, q50, q90 = np.percentile(array, [10, 50, 90])
    return f"p10={q10:.4f} median={q50:.4f} p90={q90:.4f}"


def print_sequence_report(sequence: int, observations: list[Observation]) -> None:
    print("\n" + "=" * 78)
    print(f"SEQUENCE {sequence:02d}: B1-MISSED OBJECT OBSERVABILITY")
    print("=" * 78)
    total_points = sum(item.point_count for item in observations)
    print(
        f"instance-frames={len(observations)} moving_points={total_points} "
        "(instances with >=5 points)"
    )

    print("\n---- detection groups ----")
    print(
        f"{'group':>10} {'#obs':>8} {'%obs':>8} {'points':>12} "
        f"{'%points':>9} {'med-size':>10} {'med-range':>10}"
    )
    for group in ("missed", "partial", "full"):
        selected = [item for item in observations if item.group == group]
        points = sum(item.point_count for item in selected)
        median_size = np.median([item.point_count for item in selected]) \
            if selected else 0
        median_range = np.median([item.median_range for item in selected]) \
            if selected else 0
        print(
            f"{group:>10} {len(selected):8d} "
            f"{100*len(selected)/max(len(observations),1):8.1f} "
            f"{points:12d} {100*points/max(total_points,1):9.1f} "
            f"{median_size:10.1f} {median_range:10.2f}"
        )

    missed = [item for item in observations if item.group == "missed"]
    missed_points = sum(item.point_count for item in missed)
    print("\n---- mutually exclusive causes of B1 misses ----")
    print(
        f"{'cause':>22} {'#obs':>8} {'%miss':>9} {'points':>12} "
        f"{'%misspts':>10}"
    )
    for cause in ("semantic absent", "cluster absent", "cluster low-score"):
        selected = [item for item in missed if item.failure_cause == cause]
        points = sum(item.point_count for item in selected)
        print(
            f"{cause:>22} {len(selected):8d} "
            f"{100*len(selected)/max(len(missed),1):9.1f} "
            f"{points:12d} {100*points/max(missed_points,1):10.1f}"
        )

    print("\n---- signal summaries by detection group ----")
    fields = (
        ("semantic coverage", "semantic_coverage"),
        ("cluster coverage", "cluster_coverage"),
        ("motion probability top25", "probability_top25"),
        ("motion probability max", "probability_maximum"),
        ("absolute residual top25 (m)", "residual_top25"),
        ("absolute residual max (m)", "residual_maximum"),
    )
    for label, field in fields:
        print(f"\n{label}")
        for group in ("missed", "partial", "full"):
            values = [
                float(getattr(item, field))
                for item in observations if item.group == group
            ]
            print(f"  {group:>8}: {percentile_text(values)}")

    print("\n---- fixed residual-evidence reference levels among misses ----")
    print(
        "These are descriptive metric levels in metres, not selected "
        "decision thresholds."
    )
    print(f"{'top25 residual >=':>22} {'#obs':>8} {'%miss':>9} {'%misspts':>10}")
    for level in (0.05, 0.10, 0.20, 0.50):
        selected = [item for item in missed if item.residual_top25 >= level]
        points = sum(item.point_count for item in selected)
        print(
            f"{level:22.2f} {len(selected):8d} "
            f"{100*len(selected)/max(len(missed),1):9.1f} "
            f"{100*points/max(missed_points,1):10.1f}"
        )

    print("\n---- missed instance-frames by mover class ----")
    print(
        f"{'class':>15} {'miss/total':>14} {'miss%':>9} "
        f"{'miss points':>13} {'%misspts':>10}"
    )
    classes = sorted(
        {item.raw_class for item in observations},
        key=lambda raw_class: -sum(
            item.point_count for item in observations
            if item.raw_class == raw_class
        ),
    )
    for raw_class in classes:
        class_all = [
            item for item in observations if item.raw_class == raw_class
        ]
        class_missed = [item for item in class_all if item.group == "missed"]
        points = sum(item.point_count for item in class_missed)
        print(
            f"{MOVING_NAMES.get(raw_class, str(raw_class)):>15} "
            f"{len(class_missed):5d}/{len(class_all):<8d} "
            f"{100*len(class_missed)/max(len(class_all),1):9.1f} "
            f"{points:13d} {100*points/max(missed_points,1):10.1f}"
        )

    print("\n---- missed instance-frames by median range ----")
    print(
        f"{'range(m)':>12} {'miss/total':>14} {'miss%':>9} "
        f"{'miss points':>13} {'%misspts':>10}"
    )
    for low, high in RANGE_BINS:
        range_all = [
            item for item in observations if low <= item.median_range < high
        ]
        range_missed = [item for item in range_all if item.group == "missed"]
        points = sum(item.point_count for item in range_missed)
        print(
            f"{low:4d}-{high:<7d} {len(range_missed):5d}/{len(range_all):<8d} "
            f"{100*len(range_missed)/max(len(range_all),1):9.1f} "
            f"{points:13d} {100*points/max(missed_points,1):10.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cluster-threshold", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

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
    if not model_cfg.get("use_cluster", False):
        raise ValueError("config must enable the learned cluster head")
    if dataset_cfg.get("feat_rep") != "residual":
        raise ValueError("audit expects the frozen label-free residual setup")

    dataset = SemanticKITTI4D(
        dataset_cfg["root"], sequences, dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"], dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg["seed"], dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        frame_offsets=dataset_cfg.get("frame_offsets"),
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
    observations: dict[int, list[Observation]] = {
        sequence: [] for sequence in sequences
    }
    moving_without_instance = defaultdict(int)
    residual_count = dataset_cfg["n_frames"] - 1

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            output = model(to_st(batch, device))
            voxel_probability = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            semantic_prediction = output["semantic_logits"].argmax(1).cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            cluster_score = pooled_scores(
                voxel_probability, cluster_ids, "top25"
            )

            point_voxel = batch["ref_point_voxel"].numpy()
            point_probability = voxel_probability[point_voxel]
            point_semantic = semantic_prediction[point_voxel]
            point_cluster = cluster_ids[point_voxel]
            completed_probability = complete_top25(
                point_probability, point_cluster, cluster_score
            )
            b1_prediction = completed_probability >= args.cluster_threshold

            voxel_features = batch["feats"].numpy()
            residual_features = voxel_features[:, 1:1 + residual_count]
            point_residual = np.abs(residual_features[point_voxel]).max(axis=1)

            sequence_dir = os.path.join(
                dataset_cfg["root"], f"{sequence:02d}"
            )
            scan = _read_scan(
                os.path.join(sequence_dir, "velodyne", f"{frame:06d}.bin")
            )
            xyz = scan[:, :3]
            keep = np.all(
                np.abs(xyz) < dataset_cfg["point_range"], axis=1
            )
            semantic_raw, instance_raw = split_label(
                _read_label(
                    os.path.join(
                        sequence_dir, "labels", f"{frame:06d}.label"
                    )
                )
            )
            if len(semantic_raw) != len(xyz) or int(keep.sum()) != len(
                point_voxel
            ):
                raise RuntimeError(
                    f"reference-point alignment failed at {sequence:02d}/{frame:06d}"
                )
            xyz = xyz[keep]
            semantic_raw = semantic_raw[keep]
            instance_raw = instance_raw[keep]
            moving = np.isin(semantic_raw, list(MOVING_NAMES))
            moving_without_instance[sequence] += int(
                (moving & (instance_raw <= 0)).sum()
            )

            keys = np.unique(
                semantic_raw[moving & (instance_raw > 0)].astype(np.int64)
                * 100000
                + instance_raw[moving & (instance_raw > 0)].astype(np.int64)
            )
            for key in keys:
                raw_class = int(key // 100000)
                instance_id = int(key % 100000)
                selected = (
                    moving
                    & (semantic_raw == raw_class)
                    & (instance_raw == instance_id)
                )
                point_count = int(selected.sum())
                if point_count < 5:
                    continue
                probabilities = point_probability[selected]
                residuals = point_residual[selected]
                observations[sequence].append(
                    Observation(
                        sequence=sequence,
                        frame=frame,
                        raw_class=raw_class,
                        instance_id=instance_id,
                        point_count=point_count,
                        median_range=float(
                            np.median(np.linalg.norm(xyz[selected], axis=1))
                        ),
                        b1_fraction=float(b1_prediction[selected].mean()),
                        semantic_coverage=float(
                            np.isin(
                                point_semantic[selected], MOVABLE_LEARNING_IDS
                            ).mean()
                        ),
                        cluster_coverage=float(
                            (point_cluster[selected] >= 0).mean()
                        ),
                        probability_mean=float(probabilities.mean()),
                        probability_top25=top_fraction(probabilities),
                        probability_maximum=float(probabilities.max()),
                        residual_mean=float(residuals.mean()),
                        residual_top25=top_fraction(residuals),
                        residual_maximum=float(residuals.max()),
                    )
                )

            if frame_index % 200 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)
            if args.max_frames and frame_index + 1 >= args.max_frames:
                break

    print(
        f"\nFrozen B1 protocol: Top25 threshold={args.cluster_threshold:.5f}; "
        "no threshold search. GT is diagnostics-only."
    )
    for sequence in sequences:
        print_sequence_report(sequence, observations[sequence])
        print(
            "moving points excluded from instance-frame analysis because "
            f"instance_id<=0: {moving_without_instance[sequence]}"
        )

    print("\n" + "=" * 78)
    print("CROSS-SEQUENCE FAILURE-CAUSE SUMMARY")
    print("=" * 78)
    print(
        f"{'sequence':>10} {'missed':>9} {'semantic absent':>17} "
        f"{'cluster absent':>16} {'cluster low':>13}"
    )
    for sequence in sequences:
        missed = [
            item for item in observations[sequence] if item.group == "missed"
        ]
        cause_counts = defaultdict(int)
        for item in missed:
            cause_counts[item.failure_cause] += 1
        print(
            f"{sequence:10d} {len(missed):9d} "
            f"{cause_counts['semantic absent']:17d} "
            f"{cause_counts['cluster absent']:16d} "
            f"{cause_counts['cluster low-score']:13d}"
        )
    print(
        "Architecture decision: semantic-absent misses require a proposal "
        "source independent of semantic foreground; cluster-absent misses "
        "require connectivity/proposal repair; cluster-low-score misses "
        "require motion scoring or training changes."
    )


if __name__ == "__main__":
    main()
