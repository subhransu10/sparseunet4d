"""Export the calibrated MapMOS + SparseUNet4D union as KITTI labels.

The frozen validation operating point is MapMOS belief OR a 50/50 v16+v15
probability ensemble with mean learned-cluster completion at 0.99995.  Each
frame is processed and written atomically, so memory remains bounded and an
interrupted run never leaves a partially written label file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from experiments.mapmos_sparse_fusion_eval import (POOL_MODES,
                                                    mapmos_motion_mask)
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import MOVING_IDS
from sparseunet4d.models.backend import backend


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_to_file(values, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.asarray(values, dtype=np.uint32).tofile(tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--mapmos-pred-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--pool", choices=POOL_MODES, default="mean")
    ap.add_argument("--threshold", type=float, default=0.99995)
    ap.add_argument("--mapmos-moving-id", type=int, default=251,
                    help="moving label in MapMOS files; use -1 for auto")
    ap.add_argument("--output-moving-id", type=int, default=251)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace matching label files already in output-dir")
    args = ap.parse_args()

    if not 0 <= args.weight_a <= 1:
        raise ValueError("--weight-a must be in [0, 1]")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if not 0 <= args.output_moving_id <= 0xFFFF:
        raise ValueError("--output-moving-id must fit in uint16")
    if backend() != "me":
        raise RuntimeError("export requires SU4D_BACKEND=me")

    config_path = Path(args.config).expanduser().resolve()
    ckpt_a = Path(args.ckpt_a).expanduser().resolve()
    ckpt_b = Path(args.ckpt_b).expanduser().resolve()
    mapmos_dir = Path(args.mapmos_pred_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == mapmos_dir:
        raise ValueError("output-dir must differ from mapmos-pred-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.label"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} label files; "
            "choose an empty directory or pass --overwrite")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    d, p, m = cfg["dataset"], cfg["pose"], cfg["model"]
    if args.pool != "point" and not m.get("use_cluster", False):
        raise ValueError("non-point pooling requires model.use_cluster=true")
    if len(d["val_sequences"]) != 1:
        raise ValueError("export currently requires exactly one val sequence")

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
    ma = load_model(str(ckpt_a), d, m, dev)
    mb = load_model(str(ckpt_b), d, m, dev)

    moving_id = (None if args.mapmos_moving_id < 0
                 else args.mapmos_moving_id)
    w = np.float32(args.weight_a)
    threshold = np.float32(args.threshold)
    tp = fp = fn = ignored_positive = added = written_points = 0
    written_names = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            seq, frame = batch["meta"][0]
            seq_dir = Path(d["root"]) / f"{seq:02d}"
            scan_path = seq_dir / "velodyne" / f"{frame:06d}.bin"
            gt_path = seq_dir / "labels" / f"{frame:06d}.label"
            mapmos_path = mapmos_dir / f"{frame:06d}.label"
            if not mapmos_path.is_file():
                raise FileNotFoundError(mapmos_path)

            scan = np.fromfile(scan_path, np.float32).reshape(-1, 4)
            gt_raw = np.fromfile(gt_path, np.uint32)
            mapmos_raw = np.fromfile(mapmos_path, np.uint32)
            if not (len(scan) == len(gt_raw) == len(mapmos_raw)):
                raise RuntimeError(
                    f"frame {frame:06d} point mismatch: scan={len(scan)} "
                    f"gt={len(gt_raw)} mapmos={len(mapmos_raw)}")

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
                    f"range points={int(in_range.sum())}, "
                    f"scores={len(clipped_score)}")

            sparse_add = np.zeros(len(scan), dtype=bool)
            sparse_add[in_range] = clipped_score >= threshold
            pred_sem = mapmos_raw & np.uint32(0xFFFF)
            mapmos_moving = mapmos_motion_mask(pred_sem, moving_id)
            fused_moving = mapmos_moving | sparse_add
            added += int((~mapmos_moving & sparse_add).sum())

            # Preserve any upper instance bits and all nonmoving MapMOS labels;
            # normalize every fused moving prediction to output-moving-id.
            fused_raw = mapmos_raw.copy()
            upper = fused_raw[fused_moving] & np.uint32(0xFFFF0000)
            fused_raw[fused_moving] = upper | np.uint32(args.output_moving_id)
            output_path = output_dir / f"{frame:06d}.label"
            atomic_to_file(fused_raw, output_path)
            written_names.append(output_path.name)
            written_points += len(fused_raw)

            gt_sem = gt_raw & np.uint32(0xFFFF)
            gt_moving = np.isin(gt_sem, list(MOVING_IDS))
            valid = (gt_sem != 0) & (gt_sem != 1)
            tp += int((valid & fused_moving & gt_moving).sum())
            fp += int((valid & fused_moving & ~gt_moving).sum())
            fn += int((valid & ~fused_moving & gt_moving).sum())
            ignored_positive += int((~valid & fused_moving).sum())

            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    actual_files = sorted(path.name for path in output_dir.glob("*.label"))
    if actual_files != sorted(written_names):
        extras = sorted(set(actual_files) - set(written_names))
        raise RuntimeError(
            f"output directory contains unexpected label files: {extras[:5]}")

    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    manifest = {
        "method": "mapmos_or_sparseunet_probability_mean_cluster",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_a": str(ckpt_a),
        "checkpoint_a_sha256": sha256_file(ckpt_a),
        "checkpoint_b": str(ckpt_b),
        "checkpoint_b_sha256": sha256_file(ckpt_b),
        "mapmos_prediction_dir": str(mapmos_dir),
        "sequence": int(d["val_sequences"][0]),
        "weight_a": args.weight_a,
        "pool": args.pool,
        "threshold": float(threshold),
        "mapmos_moving_id": args.mapmos_moving_id,
        "output_moving_id": args.output_moving_id,
        "frames": len(written_names),
        "points": written_points,
        "added_predictions": added,
        "ignored_positive_predictions": ignored_positive,
        "official_metrics": {
            "tp": tp, "fp": fp, "fn": fn, "iou": iou,
            "precision": precision, "recall": recall, "f1": f1,
        },
    }
    manifest_path = output_dir / "fusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== exported MapMOS + SparseUNet fusion ===")
    print(f"output: {output_dir}")
    print(f"frames={len(written_names)} points={written_points} "
          f"new moving predictions={added}")
    print(f"TP={tp} FP={fp} FN={fn} ignored predictions={ignored_positive}")
    print(f"IoU={iou:.6f} P={precision:.6f} R={recall:.6f} F1={f1:.6f}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
