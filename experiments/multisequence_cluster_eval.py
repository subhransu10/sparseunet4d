#!/usr/bin/env python3
"""Robust shared-threshold cluster-completion evaluation across sequences."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import MODES, pooled_scores
from experiments.ensemble_eval import load_model, threshold_counts, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate


DEFAULT_THRESHOLDS = [
    0.001, 0.002, 0.003, 0.005, 0.0075, 0.01,
    0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS
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
    n_modes = len(MODES)
    counts = {
        sequence: {
            "tp": np.zeros((n_modes, len(thresholds)), np.int64),
            "fp": np.zeros((n_modes, len(thresholds)), np.int64),
            "positive": 0,
        }
        for sequence in sequences
    }

    dataset = SemanticKITTI4D(
        dataset_cfg["root"],
        sequences,
        dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"],
        dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg["seed"],
        dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        frame_offsets=dataset_cfg.get("frame_offsets"),
        feat_rep=dataset_cfg.get("feat_rep", "label"),
        residual_validity=dataset_cfg.get("residual_validity", False),
        residual_all_frames=dataset_cfg.get("residual_all_frames", False),
        return_point_map=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=me_collate,
        num_workers=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, dataset_cfg, model_cfg, device)

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence = int(batch["meta"][0][0])
            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            learned = output.get("cluster_logits")
            learned = (
                torch.sigmoid(learned).cpu().numpy()
                if learned is not None
                else None
            )

            point_voxel = batch["ref_point_voxel"].numpy()
            ground_truth = batch["ref_point_motion"].numpy()
            valid = ground_truth != -1
            positive = ground_truth[valid] == 1
            counts[sequence]["positive"] += int(positive.sum())
            point_score = voxel_score[point_voxel][valid]
            point_cluster = cluster_ids[point_voxel][valid]

            candidates = [point_score]
            for mode in MODES[1:]:
                cluster_score = pooled_scores(
                    voxel_score,
                    cluster_ids,
                    mode,
                    learned=learned,
                )
                adjusted = point_score.copy()
                member = point_cluster >= 0
                adjusted[member] = np.maximum(
                    adjusted[member],
                    cluster_score[point_cluster[member]],
                )
                candidates.append(adjusted)

            for mode_index, score in enumerate(candidates):
                tp, fp = threshold_counts(score, positive, thresholds)
                counts[sequence]["tp"][mode_index] += tp
                counts[sequence]["fp"][mode_index] += fp

            if frame_index % 200 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)

    print(f"\n=== robust cluster completion, sequences {sequences} ===")
    sequence_header = " ".join(
        f"seq{sequence:02d}" for sequence in sequences
    )
    print(
        f"{'pool':>10} {'worst':>8} {'macro':>8} {'pooled':>8} "
        f"{'@th':>8} {'Prec':>8} {'Rec':>8} {sequence_header}"
    )

    rows = []
    for mode_index, mode in enumerate(MODES):
        sequence_iou = []
        pooled_tp = np.zeros(len(thresholds), np.int64)
        pooled_fp = np.zeros(len(thresholds), np.int64)
        pooled_positive = 0
        for sequence in sequences:
            tp = counts[sequence]["tp"][mode_index]
            fp = counts[sequence]["fp"][mode_index]
            positive = counts[sequence]["positive"]
            fn = positive - tp
            sequence_iou.append(tp / np.maximum(tp + fp + fn, 1))
            pooled_tp += tp
            pooled_fp += fp
            pooled_positive += positive

        iou_matrix = np.stack(sequence_iou)
        worst = iou_matrix.min(axis=0)
        macro = iou_matrix.mean(axis=0)
        pooled_fn = pooled_positive - pooled_tp
        pooled_iou = pooled_tp / np.maximum(
            pooled_tp + pooled_fp + pooled_fn, 1
        )
        precision = pooled_tp / np.maximum(pooled_tp + pooled_fp, 1)
        recall = pooled_tp / np.maximum(pooled_tp + pooled_fn, 1)
        candidates = np.flatnonzero(worst == worst.max())
        best = int(candidates[np.argmax(macro[candidates])])
        sequence_text = " ".join(
            f"{iou_matrix[row, best]:7.4f}"
            for row in range(len(sequences))
        )
        print(
            f"{mode:>10} {worst[best]:8.4f} {macro[best]:8.4f} "
            f"{pooled_iou[best]:8.4f} {thresholds[best]:8.5f} "
            f"{precision[best]:8.4f} {recall[best]:8.4f} "
            f"{sequence_text}"
        )
        rows.append((float(worst[best]), mode, float(thresholds[best])))

    winner = max(rows, key=lambda row: row[0])
    print(
        f"\nBEST ROBUST POOL: mode={winner[1]} "
        f"worst_IoU={winner[0]:.4f} threshold={winner[2]:.5f}"
    )


if __name__ == "__main__":
    main()
