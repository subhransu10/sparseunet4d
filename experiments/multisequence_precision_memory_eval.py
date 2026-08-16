#!/usr/bin/env python3
"""Precision-controlled causal memory on robust Top25 cluster evidence.

This evaluator is deliberately conservative.  It starts from the frozen B1
Top25 cluster completion and permits memory to *only* raise a current cluster
score when all of the following label-free conditions hold:

* a track from an earlier frame is associated by odometry in world space;
* semantic class and object size are consistent;
* the historical cluster was sufficiently confident;
* the current cluster still contains a minimum amount of motion evidence;
* the increase is convex and capped.

One setting and one decision threshold are selected jointly on all development
sequences by worst-sequence IoU.  Candidates are rejected when they exceed a
declared pooled-precision loss budget or regress any development sequence from
the frozen B1 reference.  Ground truth is used only for metric accumulation.
Audited sequence 08 is explicitly forbidden.
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
from experiments.ensemble_eval import load_model, threshold_counts, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate


DEFAULT_THRESHOLDS = [
    0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30,
    0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
]


def _world_point(pose: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (
        pose[:3, :3] @ xyz.astype(np.float64) + pose[:3, 3]
    ).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class Proposal:
    cluster_id: int
    centroid_world: np.ndarray
    radius: float
    semantic_class: int
    score: float


@dataclasses.dataclass
class Track:
    track_id: int
    centroid_world: np.ndarray
    velocity_world: np.ndarray
    radius: float
    semantic_class: int
    belief: float
    last_frame: int
    hits: int = 1

    def predicted_centroid(self, frame: int) -> np.ndarray:
        delta = max(0, int(frame) - int(self.last_frame))
        return self.centroid_world + self.velocity_world * delta


@dataclasses.dataclass(frozen=True)
class PriorEvidence:
    matched: bool
    prior_score: float
    prior_hits: int
    distance_m: float
    size_ratio: float
    class_consistent: bool


def build_top25_proposals(
    voxel_score: np.ndarray,
    cluster_ids: np.ndarray,
    semantic_pred: np.ndarray,
    coords5: np.ndarray,
    pose: np.ndarray,
    voxel_size: float,
) -> list[Proposal]:
    """Build reference-frame proposals scored by frozen Top25 pooling."""
    scores = pooled_scores(voxel_score, cluster_ids, "top25")
    proposals: list[Proposal] = []
    for cluster_id in np.unique(cluster_ids[cluster_ids >= 0]):
        rows = np.flatnonzero(cluster_ids == cluster_id)
        xyz_sensor = (
            coords5[rows, 1:4].astype(np.float32) + 0.5
        ) * voxel_size
        centroid_sensor = xyz_sensor.mean(axis=0)
        centroid_world = _world_point(pose, centroid_sensor)
        radial = np.linalg.norm(xyz_sensor - centroid_sensor, axis=1)
        radius = float(max(np.percentile(radial, 90), voxel_size))
        classes, counts = np.unique(semantic_pred[rows], return_counts=True)
        semantic_class = int(classes[int(np.argmax(counts))])
        proposals.append(
            Proposal(
                cluster_id=int(cluster_id),
                centroid_world=centroid_world,
                radius=radius,
                semantic_class=semantic_class,
                score=float(scores[int(cluster_id)]),
            )
        )
    return proposals


class StrictCausalTracker:
    """Associate proposals causally and expose the prior before updating it."""

    def __init__(
        self,
        association_gate: float,
        max_age: int,
        max_size_ratio: float,
        velocity_alpha: float,
        history_alpha: float,
    ):
        if association_gate <= 0:
            raise ValueError("association_gate must be positive")
        if max_age < 0:
            raise ValueError("max_age must be nonnegative")
        if max_size_ratio < 1:
            raise ValueError("max_size_ratio must be at least one")
        if not 0 <= velocity_alpha <= 1:
            raise ValueError("velocity_alpha must be in [0, 1]")
        if not 0 <= history_alpha < 1:
            raise ValueError("history_alpha must be in [0, 1)")
        self.association_gate = float(association_gate)
        self.max_age = int(max_age)
        self.max_size_ratio = float(max_size_ratio)
        self.velocity_alpha = float(velocity_alpha)
        self.history_alpha = float(history_alpha)
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 0
        self.sequence: int | None = None

    def reset(self, sequence: int) -> None:
        self.tracks.clear()
        self.next_track_id = 0
        self.sequence = int(sequence)

    def _new_track(self, proposal: Proposal, frame: int) -> None:
        track = Track(
            track_id=self.next_track_id,
            centroid_world=proposal.centroid_world.copy(),
            velocity_world=np.zeros(3, np.float32),
            radius=proposal.radius,
            semantic_class=proposal.semantic_class,
            belief=proposal.score,
            last_frame=int(frame),
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def update(
        self,
        sequence: int,
        frame: int,
        proposals: list[Proposal],
    ) -> dict[int, PriorEvidence]:
        """Return prior-only evidence, then causally update the track state."""
        if self.sequence != sequence:
            self.reset(sequence)
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if frame - track.last_frame <= self.max_age
        }
        tracks = list(self.tracks.values())
        matches: list[tuple[int, int, float, float]] = []

        if tracks and proposals:
            large = 1e6
            cost = np.full((len(tracks), len(proposals)), large, np.float64)
            distance = np.full_like(cost, np.inf)
            size_ratio = np.full_like(cost, np.inf)
            for track_index, track in enumerate(tracks):
                predicted = track.predicted_centroid(frame)
                for proposal_index, proposal in enumerate(proposals):
                    dist = float(
                        np.linalg.norm(predicted - proposal.centroid_world)
                    )
                    ratio = max(
                        (track.radius + 1e-3) / (proposal.radius + 1e-3),
                        (proposal.radius + 1e-3) / (track.radius + 1e-3),
                    )
                    distance[track_index, proposal_index] = dist
                    size_ratio[track_index, proposal_index] = ratio
                    if (
                        track.semantic_class == proposal.semantic_class
                        and dist <= self.association_gate
                        and ratio <= self.max_size_ratio
                    ):
                        cost[track_index, proposal_index] = (
                            dist + 0.25 * abs(np.log(ratio))
                        )
            rows, columns = linear_sum_assignment(cost)
            for track_index, proposal_index in zip(
                rows.tolist(), columns.tolist()
            ):
                if cost[track_index, proposal_index] < large:
                    matches.append(
                        (
                            track_index,
                            proposal_index,
                            float(distance[track_index, proposal_index]),
                            float(size_ratio[track_index, proposal_index]),
                        )
                    )

        evidence: dict[int, PriorEvidence] = {}
        matched_proposals: set[int] = set()
        for track_index, proposal_index, distance_m, size_ratio in matches:
            track = tracks[track_index]
            proposal = proposals[proposal_index]
            evidence[proposal.cluster_id] = PriorEvidence(
                matched=True,
                prior_score=track.belief,
                prior_hits=track.hits,
                distance_m=distance_m,
                size_ratio=size_ratio,
                class_consistent=True,
            )

            delta = max(1, int(frame) - int(track.last_frame))
            observed_velocity = (
                proposal.centroid_world - track.centroid_world
            ) / delta
            track.velocity_world = (
                self.velocity_alpha * track.velocity_world
                + (1 - self.velocity_alpha) * observed_velocity
            ).astype(np.float32)
            track.centroid_world = proposal.centroid_world.copy()
            track.radius = 0.5 * track.radius + 0.5 * proposal.radius
            track.semantic_class = proposal.semantic_class
            track.belief = (
                self.history_alpha * track.belief
                + (1 - self.history_alpha) * proposal.score
            )
            track.last_frame = int(frame)
            track.hits += 1
            matched_proposals.add(proposal_index)

        for proposal_index, proposal in enumerate(proposals):
            if proposal_index not in matched_proposals:
                self._new_track(proposal, frame)
                evidence[proposal.cluster_id] = PriorEvidence(
                    matched=False,
                    prior_score=0.0,
                    prior_hits=0,
                    distance_m=float("inf"),
                    size_ratio=float("inf"),
                    class_consistent=False,
                )
        return evidence


def controlled_cluster_scores(
    proposals: list[Proposal],
    evidence: dict[int, PriorEvidence],
    memory_weight: float,
    evidence_floor: float,
    history_threshold: float,
    minimum_history_hits: int,
    maximum_boost: float,
) -> dict[int, float]:
    """Apply a bounded prior boost; never suppress current Top25 evidence."""
    if not 0 <= memory_weight <= 1:
        raise ValueError("memory_weight must be in [0, 1]")
    if maximum_boost < 0:
        raise ValueError("maximum_boost must be nonnegative")
    adjusted: dict[int, float] = {}
    for proposal in proposals:
        current = proposal.score
        prior = evidence[proposal.cluster_id]
        score = current
        if (
            prior.matched
            and prior.class_consistent
            and prior.prior_hits >= minimum_history_hits
            and prior.prior_score >= history_threshold
            and current >= evidence_floor
            and prior.prior_score > current
        ):
            convex = current + memory_weight * (
                prior.prior_score - current
            )
            score = min(convex, current + maximum_boost)
        adjusted[proposal.cluster_id] = float(max(current, score))
    return adjusted


def cluster_complete(
    point_score: np.ndarray,
    point_cluster: np.ndarray,
    cluster_scores: dict[int, float],
) -> np.ndarray:
    adjusted = point_score.copy()
    member = point_cluster >= 0
    if member.any() and cluster_scores:
        maximum_id = max(cluster_scores)
        dense = np.zeros(maximum_id + 1, np.float32)
        for cluster_id, score in cluster_scores.items():
            dense[cluster_id] = score
        valid_member = member & (point_cluster <= maximum_id)
        adjusted[valid_member] = np.maximum(
            adjusted[valid_member], dense[point_cluster[valid_member]]
        )
    return adjusted


def metric_curves(tp: np.ndarray, fp: np.ndarray, positive: int):
    fn = positive - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    return iou, precision, recall


def robust_row(counts, sequences, row, thresholds):
    sequence_iou = []
    pooled_tp = np.zeros(len(thresholds), np.int64)
    pooled_fp = np.zeros(len(thresholds), np.int64)
    pooled_positive = 0
    for sequence in sequences:
        item = counts[sequence]
        iou, _, _ = metric_curves(
            item["tp"][row], item["fp"][row], item["positive"]
        )
        sequence_iou.append(iou)
        pooled_tp += item["tp"][row]
        pooled_fp += item["fp"][row]
        pooled_positive += item["positive"]
    matrix = np.stack(sequence_iou)
    worst = matrix.min(axis=0)
    macro = matrix.mean(axis=0)
    pooled_iou, precision, recall = metric_curves(
        pooled_tp, pooled_fp, pooled_positive
    )
    tied = np.flatnonzero(worst == worst.max())
    best = int(tied[np.argmax(macro[tied])])
    return {
        "worst": float(worst[best]),
        "macro": float(macro[best]),
        "pooled": float(pooled_iou[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(thresholds[best]),
        "sequence_iou": matrix[:, best].astype(float).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--memory-weights", type=float, nargs="+", default=[0.25, 0.50]
    )
    parser.add_argument(
        "--association-gates", type=float, nargs="+", default=[0.75, 1.25]
    )
    parser.add_argument(
        "--evidence-floors", type=float, nargs="+", default=[0.05, 0.10]
    )
    parser.add_argument(
        "--minimum-history-hits", type=int, nargs="+", default=[1, 2]
    )
    parser.add_argument("--history-threshold", type=float, default=0.20)
    parser.add_argument("--maximum-boost", type=float, default=0.15)
    parser.add_argument("--max-age", type=int, default=1)
    parser.add_argument("--max-size-ratio", type=float, default=1.5)
    parser.add_argument("--velocity-alpha", type=float, default=0.5)
    parser.add_argument("--history-alpha", type=float, default=0.5)
    parser.add_argument("--maximum-precision-drop", type=float, default=0.02)
    parser.add_argument(
        "--minimum-iou-gain",
        type=float,
        default=0.001,
        help="minimum worst-sequence IoU gain required to accept memory",
    )
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
        raise ValueError("audited sequence 08 is forbidden for model selection")
    if not model_cfg.get("use_cluster", False):
        raise ValueError("config must enable the learned cluster head")

    thresholds = np.asarray(sorted(set(args.thresholds)), np.float32)
    gates = sorted(set(float(x) for x in args.association_gates))
    settings = [
        (weight, gate, floor, hits)
        for weight in args.memory_weights
        for gate in gates
        for floor in args.evidence_floors
        for hits in args.minimum_history_hits
    ]
    labels = [("point", None), ("top25", None)] + [
        ("memory", setting) for setting in settings
    ]
    counts = {
        sequence: {
            "tp": np.zeros((len(labels), len(thresholds)), np.int64),
            "fp": np.zeros((len(labels), len(thresholds)), np.int64),
            "positive": 0,
        }
        for sequence in sequences
    }

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
    trackers = {
        gate: StrictCausalTracker(
            association_gate=gate,
            max_age=args.max_age,
            max_size_ratio=args.max_size_ratio,
            velocity_alpha=args.velocity_alpha,
            history_alpha=args.history_alpha,
        )
        for gate in gates
    }
    previous_meta: tuple[int, int] | None = None

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            if previous_meta is not None and sequence == previous_meta[0]:
                if frame != previous_meta[1] + 1:
                    raise RuntimeError(
                        f"non-causal order: {previous_meta} -> {(sequence, frame)}"
                    )
            previous_meta = (sequence, frame)

            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            semantic_pred = output["semantic_logits"].argmax(1).cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            coordinates = output["coords"].detach().cpu().numpy()
            pose = dataset.pose_providers[sequence].pose(frame)
            proposals = build_top25_proposals(
                voxel_score, cluster_ids, semantic_pred, coordinates, pose,
                float(dataset_cfg["voxel_size"]),
            )
            current_scores = {
                proposal.cluster_id: proposal.score for proposal in proposals
            }
            evidence_by_gate = {
                gate: tracker.update(sequence, frame, proposals)
                for gate, tracker in trackers.items()
            }

            point_voxel = batch["ref_point_voxel"].numpy()
            ground_truth = batch["ref_point_motion"].numpy()
            valid = ground_truth != -1
            positive = ground_truth[valid] == 1
            counts[sequence]["positive"] += int(positive.sum())
            point_score = voxel_score[point_voxel][valid]
            point_cluster = cluster_ids[point_voxel][valid]
            candidate_scores = [
                point_score,
                cluster_complete(point_score, point_cluster, current_scores),
            ]
            for weight, gate, floor, hits in settings:
                scores = controlled_cluster_scores(
                    proposals,
                    evidence_by_gate[gate],
                    memory_weight=float(weight),
                    evidence_floor=float(floor),
                    history_threshold=args.history_threshold,
                    minimum_history_hits=int(hits),
                    maximum_boost=args.maximum_boost,
                )
                candidate_scores.append(
                    cluster_complete(point_score, point_cluster, scores)
                )

            for row, score in enumerate(candidate_scores):
                tp, fp = threshold_counts(score, positive, thresholds)
                counts[sequence]["tp"][row] += tp
                counts[sequence]["fp"][row] += fp
            if frame_index % 200 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)

    rows = [
        robust_row(counts, sequences, row, thresholds)
        for row in range(len(labels))
    ]
    point = rows[0]
    baseline = rows[1]
    minimum_precision = baseline["precision"] - args.maximum_precision_drop

    print(
        f"\n=== precision-controlled Top25 memory, sequences {sequences} ==="
    )
    print(
        "Prediction uses frozen network, clusters and odometry only; "
        "GT is metric-only."
    )
    print(
        f"history_th={args.history_threshold:.3f} "
        f"max_boost={args.maximum_boost:.3f} max_age={args.max_age} "
        f"size_ratio<={args.max_size_ratio:.2f} "
        f"precision_drop<={args.maximum_precision_drop:.3f} "
        f"minimum_gain>={args.minimum_iou_gain:.4f}"
    )
    sequence_header = " ".join(
        f"seq{sequence:02d}" for sequence in sequences
    )
    print(
        f"{'mode':>8} {'weight':>7} {'gate':>6} {'floor':>7} {'hits':>5} "
        f"{'worst':>8} {'macro':>8} {'pooled':>8} {'@th':>8} "
        f"{'Prec':>8} {'Rec':>8} {'ok':>4} {sequence_header}"
    )

    accepted: list[tuple[float, float, int]] = []
    for row_index, ((mode, setting), result) in enumerate(zip(labels, rows)):
        if setting is None:
            weight_text = gate_text = floor_text = hits_text = "--"
        else:
            weight, gate, floor, hits = setting
            weight_text = f"{weight:.2f}"
            gate_text = f"{gate:.2f}"
            floor_text = f"{floor:.2f}"
            hits_text = str(hits)
        no_sequence_regression = all(
            candidate >= reference - 1e-12
            for candidate, reference in zip(
                result["sequence_iou"], baseline["sequence_iou"]
            )
        )
        within_precision_budget = result["precision"] >= minimum_precision
        is_accepted = (
            mode == "memory"
            and no_sequence_regression
            and within_precision_budget
            and result["worst"] >= (
                baseline["worst"] + args.minimum_iou_gain
            )
        )
        if is_accepted:
            accepted.append((result["worst"], result["macro"], row_index))
        sequence_text = " ".join(
            f"{value:7.4f}" for value in result["sequence_iou"]
        )
        print(
            f"{mode:>8} {weight_text:>7} {gate_text:>6} "
            f"{floor_text:>7} {hits_text:>5} "
            f"{result['worst']:8.4f} {result['macro']:8.4f} "
            f"{result['pooled']:8.4f} {result['threshold']:8.5f} "
            f"{result['precision']:8.4f} {result['recall']:8.4f} "
            f"{('yes' if is_accepted else 'no'):>4} {sequence_text}"
        )

    print(
        f"\nB0 point: worst={point['worst']:.4f} "
        f"threshold={point['threshold']:.5f}"
    )
    print(
        f"B1 Top25: worst={baseline['worst']:.4f} "
        f"threshold={baseline['threshold']:.5f} "
        f"P={baseline['precision']:.4f} R={baseline['recall']:.4f}"
    )
    if not accepted:
        print(
            "DECISION: REJECT memory. No candidate improves B1 on both "
            "development sequences within the precision budget."
        )
        return

    _, _, winner_index = max(accepted, key=lambda item: (item[0], item[1]))
    winner = rows[winner_index]
    setting = labels[winner_index][1]
    assert setting is not None
    weight, gate, floor, hits = setting
    print(
        "DECISION: ACCEPT precision-controlled memory\n"
        f"  weight={weight:.2f} gate={gate:.2f} floor={floor:.2f} "
        f"minimum_history_hits={hits}\n"
        f"  worst_IoU={winner['worst']:.4f} "
        f"delta_over_B1={winner['worst'] - baseline['worst']:+.4f} "
        f"threshold={winner['threshold']:.5f}\n"
        f"  P={winner['precision']:.4f} R={winner['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
