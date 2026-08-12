"""Streaming MapMOS + SparseUNet4D fusion feasibility evaluation.

MapMOS supplies full-resolution SemanticKITTI ``.label`` predictions.  Two
SparseUNet checkpoints supply a point probability (optionally completed with
checkpoint A's learned clusters) inside the model's configured range.  The
script never caches sequence-wide logits: confusion counts for every threshold
are accumulated one frame at a time.

The official SemanticKITTI-MOS mask is applied directly to raw ground truth:
semantic ids 0 (unlabelled) and 1 (outlier) are ignored.  MapMOS remains the
prediction outside SparseUNet's range for union/addition fusion.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, threshold_counts, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import MOVING_IDS


DEFAULT_THRESHOLDS = [x / 100 for x in range(5, 100, 5)] + [0.53, 0.70, 0.97,
                                                               0.98, 0.99]
POOL_MODES = ("point", "mean", "top25", "top10", "max", "learned")


def read_u32(path):
    return np.fromfile(path, dtype=np.uint32)


def metrics(tp, fp, total_pos):
    fn = total_pos - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(total_pos, 1)
    return iou, precision, recall, fn


def best_row(name, tp, fp, total_pos, thresholds):
    iou, precision, recall, fn = metrics(tp, fp, total_pos)
    i = int(np.argmax(iou))
    return {
        "name": name,
        "i": i,
        "threshold": float(thresholds[i]),
        "tp": int(tp[i]),
        "fp": int(fp[i]),
        "fn": int(fn[i]),
        "iou": float(iou[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
    }


def print_row(row):
    print(f"{row['name']:>14} {row['iou']:9.4f} {row['threshold']:8.5f} "
          f"{row['precision']:8.4f} {row['recall']:8.4f} "
          f"{row['tp']:10d} {row['fp']:9d} {row['fn']:9d}")


def mapmos_motion_mask(pred_sem, moving_id):
    if moving_id is not None:
        return pred_sem == moving_id
    # Auto mode accepts MapMOS's 251 convention, native SemanticKITTI moving
    # ids, and binary {0,1} files.
    if pred_sem.size and int(pred_sem.max()) <= 1:
        return pred_sem == 1
    return (pred_sem == 251) | np.isin(pred_sem, list(MOVING_IDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--mapmos-pred-dir", required=True,
                    help="directory containing 000000.label ...")
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--pool", choices=POOL_MODES, default="mean")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--mapmos-moving-id", type=int, default=251,
                    help="moving label in MapMOS files; use -1 for auto")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if not 0 <= args.weight_a <= 1:
        raise ValueError("--weight-a must be in [0, 1]")
    moving_id = None if args.mapmos_moving_id < 0 else args.mapmos_moving_id
    thresholds = np.asarray(sorted(set(args.thresholds)), dtype=np.float32)
    if not len(thresholds) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("thresholds must be in [0, 1]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    d, p, m = cfg["dataset"], cfg["pose"], cfg["model"]
    if args.pool != "point" and not m.get("use_cluster", False):
        raise ValueError("non-point pooling requires model.use_cluster=true")
    if len(d["val_sequences"]) != 1:
        raise ValueError("fusion evaluation currently requires one val sequence")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"),
        feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        return_point_map=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=me_collate,
                        num_workers=args.num_workers)
    ma = load_model(args.ckpt_a, d, m, dev)
    mb = load_model(args.ckpt_b, d, m, dev)

    shape = (len(thresholds),)
    sparse_tp = np.zeros(shape, np.int64)
    sparse_fp = np.zeros(shape, np.int64)
    add_tp = np.zeros(shape, np.int64)
    add_fp = np.zeros(shape, np.int64)
    keep_tp = np.zeros(shape, np.int64)
    keep_fp = np.zeros(shape, np.int64)
    map_tp = map_fp = total_pos = ignored_map_positive = 0

    w = np.float32(args.weight_a)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            seq, frame = batch["meta"][0]
            seq_dir = os.path.join(d["root"], f"{seq:02d}")
            scan_path = os.path.join(seq_dir, "velodyne", f"{frame:06d}.bin")
            gt_path = os.path.join(seq_dir, "labels", f"{frame:06d}.label")
            pred_path = os.path.join(args.mapmos_pred_dir, f"{frame:06d}.label")
            if not os.path.isfile(pred_path):
                raise FileNotFoundError(pred_path)

            scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            gt_sem = read_u32(gt_path) & 0xFFFF
            pred_sem = read_u32(pred_path) & 0xFFFF
            if not (len(scan) == len(gt_sem) == len(pred_sem)):
                raise RuntimeError(
                    f"frame {frame:06d} point mismatch: scan={len(scan)} "
                    f"gt={len(gt_sem)} mapmos={len(pred_sem)}")

            x = to_st(batch, dev)
            oa, ob = ma(x), mb(x)
            pva = torch.softmax(oa["motion_logits"], 1)[:, 1].cpu().numpy()
            pvb = torch.softmax(ob["motion_logits"], 1)[:, 1].cpu().numpy()
            voxel_score = w * pva + (1.0 - w) * pvb
            rpv = batch["ref_point_voxel"].numpy()
            clipped_score = voxel_score[rpv]

            if args.pool != "point":
                cluster_ids = oa["cluster_row_id"].cpu().numpy()
                learned = oa.get("cluster_logits")
                learned = (torch.sigmoid(learned).cpu().numpy()
                           if learned is not None else None)
                cluster_score = pooled_scores(
                    voxel_score, cluster_ids, args.pool, learned=learned)
                point_cluster = cluster_ids[rpv]
                member = point_cluster >= 0
                clipped_score = clipped_score.copy()
                clipped_score[member] = np.maximum(
                    clipped_score[member], cluster_score[point_cluster[member]])

            point_range = d.get("point_range")
            if point_range is None:
                in_range = np.ones(len(scan), dtype=bool)
            else:
                in_range = np.all(np.abs(scan[:, :3]) < point_range, axis=1)
            if int(in_range.sum()) != len(clipped_score):
                raise RuntimeError(
                    f"frame {frame:06d} SparseUNet alignment mismatch: "
                    f"range points={int(in_range.sum())}, scores={len(clipped_score)}")

            # -inf means SparseUNet abstains outside its configured crop.
            sparse_score = np.full(len(scan), -np.inf, dtype=np.float32)
            sparse_score[in_range] = clipped_score
            map_pred = mapmos_motion_mask(pred_sem, moving_id)
            gt_moving = np.isin(gt_sem, list(MOVING_IDS))
            valid = (gt_sem != 0) & (gt_sem != 1)
            ignored_map_positive += int((~valid & map_pred).sum())

            positive = gt_moving[valid]
            mp = map_pred[valid]
            ss = sparse_score[valid]
            total_pos += int(positive.sum())
            map_tp += int((mp & positive).sum())
            map_fp += int((mp & ~positive).sum())

            bt, bf = threshold_counts(ss, positive, thresholds)
            sparse_tp += bt
            sparse_fp += bf
            # Union adds only where MapMOS currently predicts static.
            bt, bf = threshold_counts(ss[~mp], positive[~mp], thresholds)
            add_tp += bt
            add_fp += bf
            # Intersection retains only MapMOS-moving points accepted by Sparse.
            bt, bf = threshold_counts(ss[mp], positive[mp], thresholds)
            keep_tp += bt
            keep_fp += bf

            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    union_tp, union_fp = map_tp + add_tp, map_fp + add_fp
    intersection_tp, intersection_fp = keep_tp, keep_fp
    oracle_add_tp = union_tp
    oracle_add_fp = np.full(shape, map_fp, np.int64)
    oracle_veto_tp = np.full(shape, map_tp, np.int64)
    oracle_veto_fp = keep_fp
    oracle_choice_tp, oracle_choice_fp = union_tp, keep_fp

    map_fn = total_pos - map_tp
    map_iou = map_tp / max(map_tp + map_fp + map_fn, 1)
    map_p = map_tp / max(map_tp + map_fp, 1)
    map_r = map_tp / max(total_pos, 1)
    print(f"\n=== MapMOS + SparseUNet fusion, val seq {d['val_sequences']} ===")
    print(f"Sparse: w(A)={args.weight_a:.2f}, pool={args.pool}; "
          f"outside-range policy=MapMOS/abstain")
    print(f"MapMOS ignored positive predictions: {ignored_map_positive}")
    print(f"\n{'mode':>14} {'IoU':>9} {'@th':>6} {'Prec':>8} {'Rec':>8} "
          f"{'TP':>10} {'FP':>9} {'FN':>9}")
    print(f"{'mapmos':>14} {map_iou:9.4f} {'--':>6} {map_p:8.4f} "
          f"{map_r:8.4f} {map_tp:10d} {map_fp:9d} {map_fn:9d}")

    rows = [
        best_row("sparse", sparse_tp, sparse_fp, total_pos, thresholds),
        best_row("union", union_tp, union_fp, total_pos, thresholds),
        best_row("intersection", intersection_tp, intersection_fp,
                 total_pos, thresholds),
        best_row("oracle-add", oracle_add_tp, oracle_add_fp,
                 total_pos, thresholds),
        best_row("oracle-veto", oracle_veto_tp, oracle_veto_fp,
                 total_pos, thresholds),
        best_row("oracle-choice", oracle_choice_tp, oracle_choice_fp,
                 total_pos, thresholds),
    ]
    for row in rows:
        print_row(row)

    union = rows[1]
    ui = union["i"]
    intersection = rows[2]
    ii = intersection["i"]
    print("\nFusion deltas at each actual mode's best threshold:")
    print(f"  union: recovered TP={int(add_tp[ui])}, "
          f"added FP={int(add_fp[ui])}, "
          f"IoU delta={union['iou'] - map_iou:+.4f}")
    print(f"  intersection: removed FP={map_fp - int(keep_fp[ii])}, "
          f"lost TP={map_tp - int(keep_tp[ii])}, "
          f"IoU delta={intersection['iou'] - map_iou:+.4f}")
    actual_best = max([{"name": "mapmos", "iou": map_iou}] + rows[:3],
                      key=lambda row: row["iou"])
    print(f"\nBEST ACTUAL: {actual_best['name']} IoU={actual_best['iou']:.4f}")
    print(f"ORACLE CHOICE CEILING: {rows[-1]['iou']:.4f} "
          f"@ sparse threshold {rows[-1]['threshold']:.2f}")


if __name__ == "__main__":
    main()
