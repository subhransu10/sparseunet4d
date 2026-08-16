#!/usr/bin/env python3
"""Moving-vs-static cluster separability from raw temporal residual patterns.

The audit uses frozen predicted semantic clusters on development sequences and
does not alter predictions.  Ground truth assigns each predicted cluster a
diagnostic majority-motion label.  Residual statistics are then evaluated as
ranking signals using AUROC, both for all moving clusters and for *hard moving
clusters* whose frozen Top25 network probability is below 0.20.

No decision threshold is selected.  A residual-consistency head is recommended
only when at least one predefined residual statistic reaches hard-positive
AUROC >= 0.70 independently on every development sequence.  Otherwise the
residual representation itself should be redesigned.  Audited sequence 08 is
explicitly forbidden.
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
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate


FEATURES = (
    "motion_top25",
    "residual_abs_top25",
    "residual_abs_top25_normalized",
    "residual_coherent",
    "residual_coherent_normalized",
    "residual_sign_consistency",
    "residual_offset_support",
    "residual_consistency",
)
RESIDUAL_FEATURES = FEATURES[1:]


def top_fraction(values: np.ndarray, fraction: float = 0.25) -> float:
    values = np.asarray(values, np.float64).reshape(-1)
    if len(values) == 0:
        return 0.0
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, len(values) - count)[-count:].mean())


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-correct binary AUROC; return NaN when either class is absent."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)


@dataclasses.dataclass(frozen=True)
class ClusterRecord:
    sequence: int
    frame: int
    cluster_id: int
    voxel_count: int
    point_count: int
    moving_fraction: float
    median_range: float
    saturation_fraction: float
    values: dict[str, float]

    @property
    def moving(self) -> bool:
        return self.moving_fraction >= 0.50

    @property
    def hard_moving(self) -> bool:
        return self.moving and self.values["motion_top25"] < 0.20


def cluster_residual_features(
    residual: np.ndarray,
    median_range: float,
    signal_floor: float,
    clip_value: float,
) -> tuple[dict[str, float], float]:
    """Predefined magnitude, coherence, sign and offset-support statistics."""
    if residual.ndim != 2 or residual.shape[1] == 0:
        raise ValueError("residual must have shape (voxels, offsets)")
    absolute = np.abs(residual)
    per_voxel_strength = absolute.max(axis=1)
    abs_top25 = top_fraction(per_voxel_strength)
    signed_offset_mean = residual.mean(axis=0)
    coherent = float(np.abs(signed_offset_mean).max())

    sign_consistency_by_offset = []
    support_by_offset = []
    for offset in range(residual.shape[1]):
        channel = residual[:, offset]
        active = np.abs(channel) >= signal_floor
        support_by_offset.append(float(active.mean()))
        if active.any():
            sign_consistency_by_offset.append(
                float(abs(np.sign(channel[active]).mean()))
            )
        else:
            sign_consistency_by_offset.append(0.0)
    sign_consistency = float(max(sign_consistency_by_offset))
    # Fraction of temporal offsets supported by at least 25% of cluster voxels.
    offset_support = float(
        (np.asarray(support_by_offset) >= 0.25).mean()
    )
    normalized_range = max(float(median_range), 1.0)
    # This score rewards coherent direction, spatial sign agreement and
    # persistence across offsets without introducing a fitted parameter.
    consistency = coherent * sign_consistency * offset_support
    saturation = float((absolute >= clip_value - 1e-5).mean())
    return {
        "residual_abs_top25": abs_top25,
        "residual_abs_top25_normalized": abs_top25 / normalized_range,
        "residual_coherent": coherent,
        "residual_coherent_normalized": coherent / normalized_range,
        "residual_sign_consistency": sign_consistency,
        "residual_offset_support": offset_support,
        "residual_consistency": consistency,
    }, saturation


def percentile_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    q10, q50, q90 = np.percentile(np.asarray(values), [10, 50, 90])
    return f"p10={q10:.4f} median={q50:.4f} p90={q90:.4f}"


def records_for_sequence(records, sequence):
    return [record for record in records if record.sequence == sequence]


def auc_for_group(records, feature, hard_only=False):
    negative = [record for record in records if not record.moving]
    if hard_only:
        positive = [record for record in records if record.hard_moving]
    else:
        positive = [record for record in records if record.moving]
    selected = negative + positive
    labels = np.asarray([record.moving for record in selected], bool)
    scores = np.asarray([record.values[feature] for record in selected])
    return auroc(labels, scores), len(positive), len(negative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hard-threshold", type=float, default=0.20)
    parser.add_argument("--signal-floor-m", type=float, default=0.10)
    parser.add_argument("--minimum-hard-auroc", type=float, default=0.70)
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
    if dataset_cfg.get("feat_rep") != "residual":
        raise ValueError("audit expects the frozen label-free residual setup")
    if not dataset_cfg.get("residual_feats", True):
        raise ValueError("audit requires temporal residual input channels")
    if abs(args.hard_threshold - 0.20) > 1e-12:
        raise ValueError("frozen B1 hard threshold must remain 0.20")

    dataset = SemanticKITTI4D(
        dataset_cfg["root"], sequences, dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"], dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg["seed"], dataset_cfg["point_range"],
        residual_feats=True,
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
    residual_count = dataset_cfg["n_frames"] - 1
    clip_value = float(dataset_cfg.get("res_clip", 3.0))
    records: list[ClusterRecord] = []
    ignored_clusters = defaultdict(int)
    mixed_negative_clusters = defaultdict(int)

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
            coordinates = output["coords"].detach().cpu().numpy()
            voxel_features = batch["feats"].numpy()
            residual = voxel_features[:, 1:1 + residual_count]
            point_voxel = batch["ref_point_voxel"].numpy()
            point_motion = batch["ref_point_motion"].numpy()
            point_cluster = cluster_ids[point_voxel]

            for cluster_id in np.unique(cluster_ids[cluster_ids >= 0]):
                cluster_id = int(cluster_id)
                voxel_rows = np.flatnonzero(cluster_ids == cluster_id)
                point_rows = np.flatnonzero(point_cluster == cluster_id)
                valid_points = point_rows[point_motion[point_rows] != -1]
                if len(valid_points) == 0:
                    ignored_clusters[sequence] += 1
                    continue
                moving_fraction = float(
                    (point_motion[valid_points] == 1).mean()
                )
                if 0.0 < moving_fraction < 0.50:
                    mixed_negative_clusters[sequence] += 1
                xyz = (
                    coordinates[voxel_rows, 1:4].astype(np.float32) + 0.5
                ) * float(dataset_cfg["voxel_size"])
                median_range = float(
                    np.median(np.linalg.norm(xyz, axis=1))
                )
                values, saturation = cluster_residual_features(
                    residual[voxel_rows], median_range,
                    args.signal_floor_m, clip_value,
                )
                values["motion_top25"] = float(cluster_motion[cluster_id])
                records.append(
                    ClusterRecord(
                        sequence=sequence,
                        frame=frame,
                        cluster_id=cluster_id,
                        voxel_count=len(voxel_rows),
                        point_count=len(valid_points),
                        moving_fraction=moving_fraction,
                        median_range=median_range,
                        saturation_fraction=saturation,
                        values=values,
                    )
                )

            if frame_index % 200 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)
            if args.max_frames and frame_index + 1 >= args.max_frames:
                break

    print("\n=== predicted-cluster residual separability audit ===")
    print(
        f"sequences={sequences} hard_motion_top25<{args.hard_threshold:.2f} "
        f"signal_floor={args.signal_floor_m:.2f}m "
        f"required_worst_hard_AUROC={args.minimum_hard_auroc:.2f}"
    )
    print(
        "GT supplies diagnostic majority-motion labels only; no threshold or "
        "model parameter is selected."
    )

    print("\n---- cluster populations ----")
    print(
        f"{'sequence':>10} {'static':>10} {'moving':>10} {'hard mov':>10} "
        f"{'mixed-neg':>11} {'ignored':>9}"
    )
    for sequence in sequences:
        selected = records_for_sequence(records, sequence)
        static = sum(not record.moving for record in selected)
        moving = sum(record.moving for record in selected)
        hard = sum(record.hard_moving for record in selected)
        print(
            f"{sequence:10d} {static:10d} {moving:10d} {hard:10d} "
            f"{mixed_negative_clusters[sequence]:11d} "
            f"{ignored_clusters[sequence]:9d}"
        )

    print("\n---- all-moving-cluster AUROC ----")
    sequence_header = " ".join(
        f"seq{sequence:02d}" for sequence in sequences
    )
    print(f"{'feature':>36} {'worst':>8} {sequence_header}")
    full_auc = {}
    for feature in FEATURES:
        values = []
        for sequence in sequences:
            auc_value, _, _ = auc_for_group(
                records_for_sequence(records, sequence), feature
            )
            values.append(auc_value)
        full_auc[feature] = values
        finite = [value for value in values if np.isfinite(value)]
        worst = min(finite) if len(finite) == len(values) else float("nan")
        print(
            f"{feature:>36} {worst:8.4f} "
            + " ".join(f"{value:7.4f}" for value in values)
        )

    print("\n---- HARD-moving-cluster AUROC versus all static clusters ----")
    print(
        "Hard positives are moving clusters the frozen B1 network scores "
        "below 0.20."
    )
    print(f"{'feature':>36} {'worst':>8} {sequence_header}")
    hard_auc = {}
    for feature in RESIDUAL_FEATURES:
        values = []
        for sequence in sequences:
            auc_value, _, _ = auc_for_group(
                records_for_sequence(records, sequence), feature,
                hard_only=True,
            )
            values.append(auc_value)
        hard_auc[feature] = values
        finite = [value for value in values if np.isfinite(value)]
        worst = min(finite) if len(finite) == len(values) else float("nan")
        print(
            f"{feature:>36} {worst:8.4f} "
            + " ".join(f"{value:7.4f}" for value in values)
        )

    print("\n---- group distributions ----")
    for sequence in sequences:
        selected = records_for_sequence(records, sequence)
        groups = {
            "static": [record for record in selected if not record.moving],
            "hard-moving": [record for record in selected if record.hard_moving],
            "easy-moving": [
                record for record in selected
                if record.moving and not record.hard_moving
            ],
        }
        print(f"\nsequence {sequence:02d}")
        for feature in RESIDUAL_FEATURES:
            print(f"  {feature}")
            for group, group_records in groups.items():
                print(
                    f"    {group:>11}: "
                    f"{percentile_text([r.values[feature] for r in group_records])}"
                )
        print("  saturation fraction")
        for group, group_records in groups.items():
            print(
                f"    {group:>11}: "
                f"{percentile_text([r.saturation_fraction for r in group_records])}"
            )

    candidates = []
    for feature, values in hard_auc.items():
        if all(np.isfinite(value) for value in values):
            candidates.append((min(values), float(np.mean(values)), feature))
    winner = max(candidates, default=(float("nan"), float("nan"), "none"))
    print("\n=== ARCHITECTURE DECISION ===")
    if np.isfinite(winner[0]) and winner[0] >= args.minimum_hard_auroc:
        print(
            "ACCEPT residual-consistency head probe\n"
            f"best_feature={winner[2]} worst_hard_AUROC={winner[0]:.4f} "
            f"macro_hard_AUROC={winner[1]:.4f}"
        )
        if "normalized" in winner[2]:
            print(
                "The winning signal is range-normalized; the probe must use "
                "range-normalized residual evidence explicitly."
            )
    else:
        print(
            "REJECT residual-consistency head with the current input encoding\n"
            f"best_feature={winner[2]} worst_hard_AUROC={winner[0]:.4f}\n"
            "Redesign the residual representation before adding another head."
        )


if __name__ == "__main__":
    main()
