#!/usr/bin/env python3
"""Cache object-level SparseUNet/MapMOS disagreement evidence.

The cache is deliberately compact: one row per predicted SparseUNet object
cluster, not one row per LiDAR point.  Features are prediction-only and may be
used by a resolver at inference. Ground truth contributes only weighted action
outcomes: true/false additions to MapMOS and true/false MapMOS points retained
inside each object. This supports exact risk-controlled resolver training
without retaining hundreds of millions of point records.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_eval import load_model, to_st
from experiments.mapmos_sparse_fusion_eval import mapmos_motion_mask, read_u32
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import MOVING_IDS


PROBABILITY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SEMANTIC_CLASSES = 20


def top_fraction(values: np.ndarray, fraction: float = 0.25) -> float:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return 0.0
    count = max(1, int(np.ceil(len(values) * float(fraction))))
    start = len(values) - count
    return float(np.partition(values, start)[start:].mean())


def feature_names(feature_width: int) -> list[str]:
    names = [
        "log_point_count", "range_mean", "range_std", "range_p90",
        "sparse_mean", "sparse_std", "sparse_q10", "sparse_q25",
        "sparse_q50", "sparse_q75", "sparse_q90", "sparse_max",
        "sparse_top25", "mapmos_positive_fraction",
        "sparse_mean_mapmos_positive", "sparse_mean_mapmos_negative",
        "expert_absolute_disagreement", "cluster_learned_probability",
    ]
    names.extend(f"voxel_feature_abs_mean_{index}"
                 for index in range(feature_width))
    names.extend(f"semantic_fraction_{index}"
                 for index in range(SEMANTIC_CLASSES))
    return names


def cluster_feature_vector(xyz: np.ndarray, sparse_probability: np.ndarray,
                           mapmos_positive: np.ndarray,
                           point_features: np.ndarray,
                           semantic_prediction: np.ndarray,
                           learned_probability: float) -> np.ndarray:
    """Build a fixed-width, label-free feature vector for one cluster."""
    xyz = np.asarray(xyz, dtype=np.float32)
    sparse_probability = np.asarray(sparse_probability, dtype=np.float32)
    mapmos_positive = np.asarray(mapmos_positive, dtype=bool)
    point_features = np.asarray(point_features, dtype=np.float32)
    semantic_prediction = np.asarray(semantic_prediction, dtype=np.int64)
    count = len(sparse_probability)
    if count == 0:
        raise ValueError("cluster must contain at least one point")
    if not (len(xyz) == len(mapmos_positive) == len(point_features)
            == len(semantic_prediction) == count):
        raise ValueError("cluster arrays must align")
    radial = np.linalg.norm(xyz, axis=1)
    quantiles = np.quantile(sparse_probability, PROBABILITY_QUANTILES)
    positive_score = (float(sparse_probability[mapmos_positive].mean())
                      if mapmos_positive.any()
                      else float(sparse_probability.mean()))
    negative_score = (float(sparse_probability[~mapmos_positive].mean())
                      if (~mapmos_positive).any()
                      else float(sparse_probability.mean()))
    semantic_hist = np.bincount(
        semantic_prediction[(semantic_prediction >= 0)
                            & (semantic_prediction < SEMANTIC_CLASSES)],
        minlength=SEMANTIC_CLASSES).astype(np.float32)
    semantic_hist /= max(float(semantic_hist.sum()), 1.0)
    values = [
        np.log1p(count), radial.mean(), radial.std(),
        np.quantile(radial, 0.90), sparse_probability.mean(),
        sparse_probability.std(), *quantiles.tolist(),
        sparse_probability.max(), top_fraction(sparse_probability),
        mapmos_positive.mean(), positive_score, negative_score,
        np.abs(sparse_probability - mapmos_positive.astype(np.float32)).mean(),
        float(learned_probability),
    ]
    values.extend(np.abs(point_features).mean(axis=0).tolist())
    values.extend(semantic_hist.tolist())
    return np.asarray(values, dtype=np.float32)


def action_outcomes(mapmos_positive: np.ndarray, ground_truth_moving: np.ndarray,
                    valid: np.ndarray):
    """Exact point counts affected by cluster-level add/keep decisions."""
    mapmos_positive = np.asarray(mapmos_positive, dtype=bool)
    ground_truth_moving = np.asarray(ground_truth_moving, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    add = valid & ~mapmos_positive
    keep = valid & mapmos_positive
    return (
        int((add & ground_truth_moving).sum()),
        int((add & ~ground_truth_moving).sum()),
        int((keep & ground_truth_moving).sum()),
        int((keep & ~ground_truth_moving).sum()),
    )


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mapmos-root", required=True,
                        help="root containing sequence_06/sequence_data/... ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequences", type=int, nargs="+", required=True)
    parser.add_argument("--mapmos-moving-id", type=int, default=251)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int,
                        help="interface smoke-test limit; omit for real cache")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    d, p, model_cfg = cfg["dataset"], cfg["pose"], cfg["model"]
    if not model_cfg.get("use_cluster", False):
        raise ValueError("resolver cache requires model.use_cluster=true")
    requested = [int(value) for value in args.sequences]
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate sequences are not allowed")
    moving_id = None if args.mapmos_moving_id < 0 else args.mapmos_moving_id
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, d, model_cfg, device)
    checkpoint_sha256 = sha256(args.ckpt)

    for sequence in requested:
        prediction_dir = os.path.join(
            args.mapmos_root, f"sequence_{sequence:02d}",
            "sequence_data", "predictions")
        if not os.path.isdir(prediction_dir):
            raise FileNotFoundError(prediction_dir)
        dataset = SemanticKITTI4D(
            d["root"], [sequence], d["n_frames"], d["voxel_size"],
            d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
            d["point_range"], residual_feats=d.get("residual_feats", True),
            res_clip=d.get("res_clip", 3.0),
            frame_offsets=d.get("frame_offsets"),
            feat_rep=d.get("feat_rep", "label"),
            residual_validity=d.get("residual_validity", False),
            residual_all_frames=d.get("residual_all_frames", False),
            return_point_map=True)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=me_collate,
            num_workers=args.num_workers)

        rows = []
        frames = []
        cluster_numbers = []
        add_tp, add_fp, keep_tp, keep_fp = [], [], [], []
        map_tp = map_fp = total_positive = total_valid = ignored_positive = 0
        feature_width = None

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if (args.max_frames is not None
                        and batch_index >= args.max_frames):
                    break
                seq, frame = (int(value) for value in batch["meta"][0])
                seq_dir = os.path.join(d["root"], f"{seq:02d}")
                scan_path = os.path.join(
                    seq_dir, "velodyne", f"{frame:06d}.bin")
                gt_path = os.path.join(
                    seq_dir, "labels", f"{frame:06d}.label")
                pred_path = os.path.join(
                    prediction_dir, f"{frame:06d}.label")
                scan = np.fromfile(scan_path, np.float32).reshape(-1, 4)
                gt_sem = read_u32(gt_path) & 0xFFFF
                pred_sem = read_u32(pred_path) & 0xFFFF
                if not (len(scan) == len(gt_sem) == len(pred_sem)):
                    raise RuntimeError(f"point mismatch at {seq:02d}/{frame:06d}")

                output = model(to_st(batch, device))
                voxel_probability = torch.softmax(
                    output["motion_logits"], 1)[:, 1].cpu().numpy()
                cluster_ids = output["cluster_row_id"].cpu().numpy()
                semantic_voxel = output["semantic_logits"].argmax(1).cpu().numpy()
                learned = output.get("cluster_logits")
                learned = (torch.sigmoid(learned).cpu().numpy()
                           if learned is not None else np.zeros(0, np.float32))
                point_voxel = batch["ref_point_voxel"].numpy()
                point_probability = voxel_probability[point_voxel]
                point_cluster = cluster_ids[point_voxel]
                point_semantic = semantic_voxel[point_voxel]
                point_features = batch["feats"].numpy()[point_voxel]
                feature_width = point_features.shape[1]

                point_range = d.get("point_range")
                in_range = (np.ones(len(scan), bool) if point_range is None else
                            np.all(np.abs(scan[:, :3]) < point_range, axis=1))
                if int(in_range.sum()) != len(point_voxel):
                    raise RuntimeError(
                        f"Sparse alignment mismatch at {seq:02d}/{frame:06d}")
                xyz = scan[in_range, :3]
                mapmos_full = mapmos_motion_mask(pred_sem, moving_id)
                mapmos_point = mapmos_full[in_range]
                valid_full = (gt_sem != 0) & (gt_sem != 1)
                moving_full = np.isin(gt_sem, list(MOVING_IDS))
                valid_point = valid_full[in_range]
                moving_point = moving_full[in_range]
                batch_gt = batch["ref_point_motion"].numpy()
                expected_gt = np.where(valid_point, moving_point.astype(np.int64), -1)
                if not np.array_equal(batch_gt, expected_gt):
                    raise RuntimeError(
                        f"GT alignment mismatch at {seq:02d}/{frame:06d}")

                map_tp += int((mapmos_full & moving_full & valid_full).sum())
                map_fp += int((mapmos_full & ~moving_full & valid_full).sum())
                total_positive += int((moving_full & valid_full).sum())
                total_valid += int(valid_full.sum())
                ignored_positive += int((mapmos_full & ~valid_full).sum())

                for cluster_id in np.unique(point_cluster[point_cluster >= 0]):
                    member = point_cluster == cluster_id
                    learned_probability = (
                        float(learned[cluster_id])
                        if cluster_id < len(learned) else 0.0)
                    rows.append(cluster_feature_vector(
                        xyz[member], point_probability[member],
                        mapmos_point[member], point_features[member],
                        point_semantic[member], learned_probability))
                    outcomes = action_outcomes(
                        mapmos_point[member], moving_point[member],
                        valid_point[member])
                    frames.append(frame)
                    cluster_numbers.append(int(cluster_id))
                    add_tp.append(outcomes[0]); add_fp.append(outcomes[1])
                    keep_tp.append(outcomes[2]); keep_fp.append(outcomes[3])

                if batch_index % 200 == 0:
                    print(f"  sequence {sequence:02d} frame "
                          f"{batch_index}/{len(loader)}", flush=True)

        matrix = (np.stack(rows).astype(np.float32) if rows else
                  np.zeros((0, len(feature_names(feature_width))), np.float32))
        names = np.asarray(feature_names(feature_width))
        if matrix.shape[1] != len(names):
            raise RuntimeError("feature-name width mismatch")
        output_path = os.path.join(
            args.output_dir, f"dual_expert_seq{sequence:02d}.npz")
        np.savez_compressed(
            output_path,
            features=matrix,
            feature_names=names,
            frame=np.asarray(frames, np.int32),
            cluster_id=np.asarray(cluster_numbers, np.int32),
            add_tp=np.asarray(add_tp, np.int64),
            add_fp=np.asarray(add_fp, np.int64),
            keep_tp=np.asarray(keep_tp, np.int64),
            keep_fp=np.asarray(keep_fp, np.int64),
            map_tp=np.asarray(map_tp, np.int64),
            map_fp=np.asarray(map_fp, np.int64),
            total_positive=np.asarray(total_positive, np.int64),
            total_valid=np.asarray(total_valid, np.int64),
            ignored_mapmos_positive=np.asarray(ignored_positive, np.int64),
            sequence=np.asarray(sequence, np.int32),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
        )
        map_fn = total_positive - map_tp
        map_iou = map_tp / max(map_tp + map_fp + map_fn, 1)
        print(f"saved {output_path}: rows={len(matrix)} "
              f"features={matrix.shape[1]} MapMOS_IoU={map_iou:.4f}")


if __name__ == "__main__":
    main()
