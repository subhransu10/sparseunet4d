#!/usr/bin/env python3
"""Point-level threshold evaluation with one shared multi-sequence threshold."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from scripts.train import build_model
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import ST, backend


DEFAULT_GRID = [
    0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075,
    0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
]


def update_counts(counts, score, positive, thresholds):
    tp, fp, fn = counts
    for index, threshold in enumerate(thresholds):
        prediction = score >= threshold
        tp[index] += int((prediction & positive).sum())
        fp[index] += int((prediction & ~positive).sum())
        fn[index] += int((~prediction & positive).sum())


def curves(counts):
    tp, fp, fn = counts
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    return iou, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    dataset_cfg = cfg["dataset"]
    pose_cfg = cfg.get("pose", {"seed": 0})
    model_cfg = cfg["model"]
    sequences = sorted(int(x) for x in dataset_cfg["val_sequences"])
    if len(sequences) < 2:
        raise ValueError("multi-sequence evaluation requires at least two sequences")
    if 8 in sequences:
        raise ValueError("audited sequence 08 is forbidden for model selection")

    thresholds = np.asarray(sorted(set(args.grid)), np.float64)
    if len(thresholds) == 0:
        raise ValueError("threshold grid must not be empty")
    counts = {
        sequence: (
            np.zeros(len(thresholds), np.int64),
            np.zeros(len(thresholds), np.int64),
            np.zeros(len(thresholds), np.int64),
        )
        for sequence in sequences
    }

    dataset = SemanticKITTI4D(
        dataset_cfg["root"],
        sequences,
        dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"],
        dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg.get("seed", 0),
        dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        return_point_map=True,
        frame_offsets=dataset_cfg.get("frame_offsets"),
        feat_rep=dataset_cfg.get("feat_rep", "label"),
        residual_validity=dataset_cfg.get("residual_validity", False),
        residual_all_frames=dataset_cfg.get("residual_all_frames", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=me_collate,
        num_workers=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_frames = dataset_cfg.get("n_frames", 4)
    residual_channels = (n_frames - 1) * (
        2 if dataset_cfg.get("residual_validity", False) else 1
    )
    in_channels = (
        1 + residual_channels
        if dataset_cfg.get("residual_feats", True)
        else 1
    )
    model = build_model(
        model_cfg,
        in_channels,
        dataset_cfg.get("num_semantic", 20),
    ).to(device).eval()
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            coordinates = batch["coords"].to(device)
            features = batch["feats"].to(device)
            if backend() == "me":
                import MinkowskiEngine as ME
                sparse = ME.SparseTensor(features, coordinates=coordinates)
            else:
                sparse = ST(features, coordinates)
            output = model(sparse)
            voxel_probability = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].detach().cpu().numpy()
            point_probability = voxel_probability[
                batch["ref_point_voxel"].numpy()
            ]
            point_label = batch["ref_point_motion"].numpy()
            point_batch = batch["ref_point_batch"].numpy()

            for sample_index, meta in enumerate(batch["meta"]):
                sequence = int(meta[0])
                sample = point_batch == sample_index
                valid = sample & (point_label != -1)
                score = point_probability[valid]
                positive = point_label[valid] == 1
                update_counts(counts[sequence], score, positive, thresholds)

            if batch_index % 200 == 0:
                print(f"  batch {batch_index}/{len(loader)}", flush=True)

    sequence_curves = {
        sequence: curves(counts[sequence]) for sequence in sequences
    }
    iou_matrix = np.stack([
        sequence_curves[sequence][0] for sequence in sequences
    ])
    worst = iou_matrix.min(axis=0)
    macro = iou_matrix.mean(axis=0)
    pooled_counts = tuple(
        sum(counts[sequence][part] for sequence in sequences)
        for part in range(3)
    )
    pooled_iou, pooled_precision, pooled_recall = curves(pooled_counts)

    candidates = np.flatnonzero(worst == worst.max())
    best = int(candidates[np.argmax(macro[candidates])])

    sequence_header = " ".join(
        f"seq{sequence:02d}" for sequence in sequences
    )
    print(f"\n=== shared point threshold, sequences {sequences} ===")
    print(
        f"{'th':>9} {'worst':>8} {'macro':>8} {'pooled':>8} "
        f"{'prec':>8} {'rec':>8} {sequence_header}"
    )
    for index, threshold in enumerate(thresholds):
        sequence_text = " ".join(
            f"{sequence_curves[sequence][0][index]:7.4f}"
            for sequence in sequences
        )
        marker = "  <== best shared-worst" if index == best else ""
        print(
            f"{threshold:9.5f} {worst[index]:8.4f} "
            f"{macro[index]:8.4f} {pooled_iou[index]:8.4f} "
            f"{pooled_precision[index]:8.4f} {pooled_recall[index]:8.4f} "
            f"{sequence_text}{marker}"
        )

    print(
        "\nBEST SHARED-WORST: "
        f"IoU={worst[best]:.4f} threshold={thresholds[best]:.5f} "
        f"macro={macro[best]:.4f} pooled={pooled_iou[best]:.4f} "
        f"P={pooled_precision[best]:.4f} R={pooled_recall[best]:.4f}"
    )


if __name__ == "__main__":
    main()
