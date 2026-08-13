"""Causal temporal object-memory evaluation for SparseUNet4D.

The frozen network supplies per-voxel motion probabilities and semantic
foreground clusters.  This evaluator associates those proposals in the world
frame using odometry and carries a leaky object-level motion belief forward in
time.  Ground-truth labels are used only after prediction to accumulate the
official point-level MOS metrics; they never enter proposals, association, or
memory updates.

This is deliberately an evaluation probe rather than a training component. It
tests whether causal object memory addresses the whole-instance misses found
by ``diagnose_errors.py`` before a learnable memory module is built.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Iterable

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_eval import load_model, threshold_counts, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate


DEFAULT_THRESHOLDS = [
    0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005,
    0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10,
]


def _logit(prob: float | np.ndarray) -> float | np.ndarray:
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(prob) - np.log1p(-prob)


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    value = np.clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-value))


def _world_point(pose: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ xyz.astype(np.float64)
            + pose[:3, 3]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class Proposal:
    cluster_id: int
    centroid_world: np.ndarray
    radius: float
    semantic_class: int
    score: float
    voxel_count: int


@dataclasses.dataclass
class Track:
    track_id: int
    centroid_world: np.ndarray
    velocity_world: np.ndarray
    radius: float
    semantic_class: int
    state_logit: float
    last_frame: int
    hits: int = 1

    def predicted_centroid(self, frame: int) -> np.ndarray:
        dt = max(0, int(frame) - int(self.last_frame))
        return self.centroid_world + self.velocity_world * dt


@dataclasses.dataclass(frozen=True)
class AssociationEvidence:
    """Label-free evidence attached to one current object proposal."""
    matched: bool
    distance_m: float
    size_log_ratio: float
    class_consistent: bool


def build_proposals(voxel_score: np.ndarray, cluster_ids: np.ndarray,
                    semantic_pred: np.ndarray, coords5: np.ndarray,
                    pose: np.ndarray, voxel_size: float) -> list[Proposal]:
    """Build one label-free proposal per nonnegative reference cluster."""
    proposals = []
    ids = np.unique(cluster_ids[cluster_ids >= 0])
    for cid in ids:
        rows = np.flatnonzero(cluster_ids == cid)
        if len(rows) == 0:
            continue
        xyz_sensor = (coords5[rows, 1:4].astype(np.float32) + 0.5) * voxel_size
        centroid_sensor = xyz_sensor.mean(axis=0)
        centroid_world = _world_point(pose, centroid_sensor)
        radial = np.linalg.norm(xyz_sensor - centroid_sensor, axis=1)
        radius = float(max(np.percentile(radial, 90), voxel_size))
        classes, counts = np.unique(semantic_pred[rows], return_counts=True)
        semantic_class = int(classes[int(np.argmax(counts))])
        proposals.append(Proposal(
            cluster_id=int(cid),
            centroid_world=centroid_world,
            radius=radius,
            semantic_class=semantic_class,
            score=float(voxel_score[rows].max()),
            voxel_count=int(len(rows)),
        ))
    return proposals


class TemporalObjectMemory:
    """Nearest-object causal memory with velocity prediction and EMA belief."""

    def __init__(self, alpha: float, association_gate: float, max_age: int,
                 velocity_alpha: float = 0.5, size_weight: float = 0.5,
                 class_mismatch_penalty: float = 1.0):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if association_gate <= 0:
            raise ValueError("association_gate must be positive")
        if max_age < 0:
            raise ValueError("max_age must be nonnegative")
        self.alpha = float(alpha)
        self.association_gate = float(association_gate)
        self.max_age = int(max_age)
        self.velocity_alpha = float(velocity_alpha)
        self.size_weight = float(size_weight)
        self.class_mismatch_penalty = float(class_mismatch_penalty)
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 0
        self.sequence = None
        self.last_evidence: dict[int, AssociationEvidence] = {}

    def reset(self, sequence=None):
        self.tracks.clear()
        self.next_track_id = 0
        self.sequence = sequence
        self.last_evidence = {}

    def _new_track(self, proposal: Proposal, frame: int) -> Track:
        track = Track(
            track_id=self.next_track_id,
            centroid_world=proposal.centroid_world.copy(),
            velocity_world=np.zeros(3, np.float32),
            radius=proposal.radius,
            semantic_class=proposal.semantic_class,
            state_logit=float(_logit(proposal.score)),
            last_frame=int(frame),
        )
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track

    def update(self, sequence: int, frame: int,
               proposals: Iterable[Proposal]) -> dict[int, float]:
        """Update state and return memory-adjusted score per current cluster."""
        proposals = list(proposals)
        self.last_evidence = {}
        if self.sequence != sequence:
            self.reset(sequence)

        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if frame - track.last_frame <= self.max_age
        }
        tracks = list(self.tracks.values())
        matches: list[tuple[int, int]] = []

        if tracks and proposals:
            cost = np.empty((len(tracks), len(proposals)), np.float64)
            distance = np.empty_like(cost)
            for ti, track in enumerate(tracks):
                predicted = track.predicted_centroid(frame)
                for pi, proposal in enumerate(proposals):
                    dist = np.linalg.norm(predicted - proposal.centroid_world)
                    size = abs(np.log((track.radius + 1e-3)
                                      / (proposal.radius + 1e-3)))
                    mismatch = track.semantic_class != proposal.semantic_class
                    distance[ti, pi] = dist
                    cost[ti, pi] = (dist + self.size_weight * size
                                    + self.class_mismatch_penalty * mismatch)
            rows, cols = linear_sum_assignment(cost)
            for ti, pi in zip(rows.tolist(), cols.tolist()):
                # A larger object receives a small, bounded allowance without
                # permitting nearby independent objects to merge freely.
                allowance = min(tracks[ti].radius + proposals[pi].radius, 2.0)
                if distance[ti, pi] <= self.association_gate + 0.5 * allowance:
                    matches.append((ti, pi))

        matched_proposals = set()
        output = {}
        for ti, pi in matches:
            track, proposal = tracks[ti], proposals[pi]
            match_distance = float(distance[ti, pi])
            size_log_ratio = float(abs(np.log(
                (track.radius + 1e-3) / (proposal.radius + 1e-3))))
            class_consistent = track.semantic_class == proposal.semantic_class
            self.last_evidence[proposal.cluster_id] = AssociationEvidence(
                matched=True,
                distance_m=match_distance,
                size_log_ratio=size_log_ratio,
                class_consistent=class_consistent,
            )
            dt = max(1, int(frame) - int(track.last_frame))
            observed_velocity = ((proposal.centroid_world - track.centroid_world)
                                 / dt)
            track.velocity_world = (
                self.velocity_alpha * track.velocity_world
                + (1.0 - self.velocity_alpha) * observed_velocity
            ).astype(np.float32)
            track.centroid_world = proposal.centroid_world.copy()
            track.radius = 0.5 * track.radius + 0.5 * proposal.radius
            track.semantic_class = proposal.semantic_class
            observation = float(_logit(proposal.score))
            track.state_logit = (self.alpha * track.state_logit
                                 + (1.0 - self.alpha) * observation)
            track.last_frame = int(frame)
            track.hits += 1
            # B2 is a conservative completion probe: memory may recover a weak
            # current object, but never suppress stronger current evidence.
            output[proposal.cluster_id] = float(_sigmoid(
                max(observation, track.state_logit)))
            matched_proposals.add(pi)

        for pi, proposal in enumerate(proposals):
            if pi in matched_proposals:
                continue
            self._new_track(proposal, frame)
            output[proposal.cluster_id] = proposal.score
            self.last_evidence[proposal.cluster_id] = AssociationEvidence(
                matched=False,
                distance_m=float("inf"),
                size_log_ratio=float("inf"),
                class_consistent=False,
            )
        return output


def reliability_filter(cluster_scores: dict[int, float],
                       evidence: dict[int, AssociationEvidence],
                       association_gate: float,
                       maximum_size_ratio: float = 2.0) -> dict[int, float]:
    """Keep scores backed by a strict, physically plausible association.

    The normal matcher permits a bounded object-radius allowance.  B4 is more
    conservative: additions must satisfy the unexpanded metric gate, preserve
    semantic class, and change object radius by no more than the declared
    multiplicative ratio.  No labels or evaluation statistics enter the rule.
    """
    if maximum_size_ratio < 1.0:
        raise ValueError("maximum_size_ratio must be at least 1")
    max_size_log = float(np.log(maximum_size_ratio))
    return {
        cluster_id: score
        for cluster_id, score in cluster_scores.items()
        if cluster_id in evidence
        and evidence[cluster_id].matched
        and evidence[cluster_id].class_consistent
        and evidence[cluster_id].distance_m <= association_gate
        and evidence[cluster_id].size_log_ratio <= max_size_log
    }


def cluster_complete(point_score: np.ndarray, point_cluster: np.ndarray,
                     cluster_scores: dict[int, float]) -> np.ndarray:
    adjusted = point_score.copy()
    for cluster_id, score in cluster_scores.items():
        member = point_cluster == cluster_id
        adjusted[member] = np.maximum(adjusted[member], score)
    return adjusted


def _best_row(tp: np.ndarray, fp: np.ndarray, total_pos: int,
              thresholds: np.ndarray):
    fn = total_pos - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    index = int(np.argmax(iou))
    tpi, fpi, fni = int(tp[index]), int(fp[index]), int(fn[index])
    return (
        float(iou[index]), float(thresholds[index]),
        tpi / max(tpi + fpi, 1), tpi / max(tpi + fni, 1),
        tpi, fpi, fni,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.50, 0.70, 0.85, 0.95])
    parser.add_argument("--association-gates", type=float, nargs="+",
                        default=[1.5, 2.5, 4.0])
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--velocity-alpha", type=float, default=0.5)
    parser.add_argument("--size-weight", type=float, default=0.5)
    parser.add_argument("--class-mismatch-penalty", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    d, p, m = cfg["dataset"], cfg["pose"], cfg["model"]
    if not m.get("use_cluster", False):
        raise ValueError("config must enable the learned cluster head")
    if len(d["val_sequences"]) != 1:
        raise ValueError("streaming evaluation currently requires one sequence")
    thresholds = np.asarray(sorted(set(args.thresholds)), np.float32)
    candidates = [(float(a), float(g)) for a in args.alphas
                  for g in args.association_gates]

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
        return_point_map=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, d, m, device)
    memories = [TemporalObjectMemory(
        alpha=alpha,
        association_gate=gate,
        max_age=args.max_age,
        velocity_alpha=args.velocity_alpha,
        size_weight=args.size_weight,
        class_mismatch_penalty=args.class_mismatch_penalty,
    ) for alpha, gate in candidates]

    # Rows: raw point baseline, current-cluster max (B1), then each B2 setting.
    tp = np.zeros((2 + len(candidates), len(thresholds)), np.int64)
    fp = np.zeros_like(tp)
    total_pos = 0
    previous_meta = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            if previous_meta is not None:
                previous_sequence, previous_frame = previous_meta
                if sequence == previous_sequence and frame != previous_frame + 1:
                    raise RuntimeError(
                        f"non-causal frame order: {previous_meta} -> {(sequence, frame)}")
            previous_meta = (sequence, frame)

            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], 1)[:, 1].cpu().numpy()
            semantic_pred = output["semantic_logits"].argmax(1).cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            coords5 = output["coords"].detach().cpu().numpy()
            pose = dataset.pose_providers[sequence].pose(frame)
            proposals = build_proposals(
                voxel_score, cluster_ids, semantic_pred, coords5, pose,
                float(d["voxel_size"]))
            current_cluster_scores = {
                proposal.cluster_id: proposal.score for proposal in proposals
            }

            ref_point_voxel = batch["ref_point_voxel"].numpy()
            gt = batch["ref_point_motion"].numpy()
            valid = gt != -1
            positive = gt[valid] == 1
            total_pos += int(positive.sum())
            point_score = voxel_score[ref_point_voxel][valid]
            point_cluster = cluster_ids[ref_point_voxel][valid]
            scores = [
                point_score,
                cluster_complete(point_score, point_cluster,
                                 current_cluster_scores),
            ]
            for memory in memories:
                memory_scores = memory.update(sequence, frame, proposals)
                scores.append(cluster_complete(
                    point_score, point_cluster, memory_scores))

            for row, candidate_score in enumerate(scores):
                batch_tp, batch_fp = threshold_counts(
                    candidate_score, positive, thresholds)
                tp[row] += batch_tp
                fp[row] += batch_fp
            if batch_index % 200 == 0:
                print(f"  frame {batch_index}/{len(loader)}", flush=True)

    print(f"\n=== causal temporal object memory, val seq {d['val_sequences']} ===")
    print("Prediction path uses network outputs + odometry only; GT is metric-only.")
    print(f"max_age={args.max_age} velocity_alpha={args.velocity_alpha:.2f} "
          f"size_weight={args.size_weight:.2f} "
          f"class_penalty={args.class_mismatch_penalty:.2f}")
    print(f"{'mode':>10} {'alpha':>7} {'gate':>7} {'IoU':>9} {'@th':>9} "
          f"{'Prec':>8} {'Rec':>8}")
    rows = []
    labels = [("point", None, None), ("max", None, None)] + [
        ("memory", alpha, gate) for alpha, gate in candidates
    ]
    for row, (mode, alpha, gate) in enumerate(labels):
        result = _best_row(tp[row], fp[row], total_pos, thresholds)
        rows.append((result[0], mode, alpha, gate, *result[1:]))
        alpha_text = "--" if alpha is None else f"{alpha:.2f}"
        gate_text = "--" if gate is None else f"{gate:.2f}"
        print(f"{mode:>10} {alpha_text:>7} {gate_text:>7} "
              f"{result[0]:9.4f} {result[1]:9.5f} "
              f"{result[2]:8.4f} {result[3]:8.4f}")

    best = max(rows, key=lambda item: item[0])
    print("\nBEST: "
          f"IoU={best[0]:.4f} mode={best[1]} alpha={best[2]} "
          f"gate={best[3]} threshold={best[4]:.5f} "
          f"P={best[5]:.4f} R={best[6]:.4f}")
    point = rows[0]
    best_memory = max((row for row in rows if row[1] == "memory"),
                      key=lambda item: item[0])
    print(f"B2 delta over point: {best_memory[0] - point[0]:+.4f}")


if __name__ == "__main__":
    main()
