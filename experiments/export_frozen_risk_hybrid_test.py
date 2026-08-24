#!/usr/bin/env python3
"""Export benchmark labels from the frozen risk-controlled expert hybrid.

No ground-truth labels are read. A label-free MapMOS prevalence gate selects
one frozen action for the complete sequence: conservative SparseUNet veto,
MapMOS abstention, or cluster-reliability-controlled recovery additions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.apply_frozen_recovery_resolver import frozen_scores
from experiments.cache_dual_expert_disagreements import (
    cluster_feature_vector,
    feature_names,
)
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from experiments.mapmos_sparse_fusion_eval import mapmos_motion_mask
from experiments.regime_gated_hybrid_eval import (
    ADD_THRESHOLD,
    DEFAULT_MARGIN_FACTOR,
    PREVALENCE_GATE,
    VETO_THRESHOLD,
    choose_regime,
    raw_motion_prevalence,
)
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_u32(values: np.ndarray, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.asarray(values, np.uint32).tofile(temporary)
    os.replace(temporary, path)


def fuse_mask(regime: str, mapmos_moving: np.ndarray,
              point_cluster: np.ndarray, selected_clusters: np.ndarray,
              sparse_keep: np.ndarray) -> np.ndarray:
    if regime == "mapmos-abstain":
        return mapmos_moving.copy()
    if regime == "conservative-veto":
        return mapmos_moving & sparse_keep
    if regime == "recovery-add":
        selected = np.zeros(len(point_cluster), bool)
        member = point_cluster >= 0
        selected[member] = selected_clusters[point_cluster[member]]
        return mapmos_moving | selected
    raise ValueError(regime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--resolver", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--mapmos-pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-factor", type=float,
                        default=DEFAULT_MARGIN_FACTOR)
    parser.add_argument("--mapmos-moving-id", type=int, default=251)
    parser.add_argument("--output-moving-id", type=int, default=251)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if backend() != "me":
        raise RuntimeError("export requires SU4D_BACKEND=me")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    resolver_path = Path(args.resolver).expanduser().resolve()
    mapmos_dir = Path(args.mapmos_pred_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == mapmos_dir:
        raise ValueError("output-dir must differ from MapMOS predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.label"))
    if existing:
        raise FileExistsError(
            f"output directory already contains {len(existing)} labels")

    prevalence, mapmos_positive, raw_points, expected_frames = (
        raw_motion_prevalence(mapmos_dir))
    regime = choose_regime(
        prevalence, PREVALENCE_GATE, args.margin_factor)
    print("=== frozen test-sequence regime ===", flush=True)
    print(f"sequence={args.sequence:02d} frames={expected_frames} "
          f"raw_points={raw_points} mapmos_positive={mapmos_positive}",
          flush=True)
    print(f"prevalence={prevalence:.8f} gate={PREVALENCE_GATE:.8f} "
          f"margin_factor={args.margin_factor:.3f} regime={regime}",
          flush=True)

    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    dataset_cfg, pose_cfg, model_cfg = (
        cfg["dataset"], cfg["pose"], cfg["model"])
    resolver = np.load(resolver_path)

    dataset = SemanticKITTI4D(
        dataset_cfg["root"], [args.sequence], dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"], dataset_cfg["semantic_yaml"],
        pose_cfg.get("mode", "gt"), pose_cfg.get("rot_std_deg", 0.0),
        pose_cfg.get("trans_std_m", 0.0), pose_cfg.get("seed", 0),
        dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        frame_offsets=dataset_cfg.get("frame_offsets"),
        feat_rep=dataset_cfg.get("feat_rep", "label"),
        residual_validity=dataset_cfg.get("residual_validity", False),
        residual_all_frames=dataset_cfg.get("residual_all_frames", False),
        return_point_map=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=me_collate,
        num_workers=args.num_workers)
    if len(loader) != expected_frames:
        raise RuntimeError(
            f"frame mismatch: Sparse={len(loader)} MapMOS={expected_frames}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        str(checkpoint_path), dataset_cfg, model_cfg, device)
    moving_id = None if args.mapmos_moving_id < 0 else args.mapmos_moving_id
    add_threshold = float(resolver["add_threshold"])
    added = removed = written_points = 0
    output_names = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            sequence, frame = (int(value) for value in batch["meta"][0])
            sequence_dir = Path(dataset_cfg["root"]) / f"{sequence:02d}"
            scan_path = sequence_dir / "velodyne" / f"{frame:06d}.bin"
            mapmos_path = mapmos_dir / f"{frame:06d}.label"
            scan = np.fromfile(scan_path, np.float32).reshape(-1, 4)
            mapmos_raw = np.fromfile(mapmos_path, np.uint32)
            if len(scan) != len(mapmos_raw):
                raise RuntimeError(f"point mismatch at {sequence:02d}/{frame:06d}")

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
            expected_feature_names = np.asarray(
                feature_names(point_features.shape[1]))
            if not np.array_equal(
                    resolver["feature_names"], expected_feature_names):
                raise ValueError(
                    "resolver and inference feature schemas differ")

            point_range = dataset_cfg.get("point_range")
            in_range = (np.ones(len(scan), bool) if point_range is None else
                        np.all(np.abs(scan[:, :3]) < point_range, axis=1))
            if int(in_range.sum()) != len(point_voxel):
                raise RuntimeError(
                    f"Sparse alignment mismatch at {sequence:02d}/{frame:06d}")
            xyz = scan[in_range, :3]
            mapmos_semantic = mapmos_raw & np.uint32(0xFFFF)
            mapmos_full = mapmos_motion_mask(mapmos_semantic, moving_id)
            mapmos_point = mapmos_full[in_range]

            cluster_count = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
            selected_clusters = np.zeros(cluster_count, bool)
            if regime == "recovery-add" and cluster_count:
                feature_rows = []
                feature_cluster_ids = []
                for cluster_id in np.unique(point_cluster[point_cluster >= 0]):
                    member = point_cluster == cluster_id
                    learned_probability = (
                        float(learned[cluster_id])
                        if cluster_id < len(learned) else 0.0)
                    feature_rows.append(cluster_feature_vector(
                        xyz[member], point_probability[member],
                        mapmos_point[member], point_features[member],
                        point_semantic[member], learned_probability))
                    feature_cluster_ids.append(int(cluster_id))
                if feature_rows:
                    matrix = np.stack(feature_rows).astype(np.float32)
                    scores = frozen_scores(matrix, resolver)
                    selected_clusters[np.asarray(feature_cluster_ids)] = (
                        scores >= add_threshold)

            top25 = pooled_scores(
                voxel_probability, cluster_ids, "top25", learned=learned)
            point_keep_score = point_probability.copy()
            member = point_cluster >= 0
            point_keep_score[member] = np.maximum(
                point_keep_score[member], top25[point_cluster[member]])
            sparse_keep = point_keep_score >= np.float32(VETO_THRESHOLD)
            fused_point = fuse_mask(
                regime, mapmos_point, point_cluster,
                selected_clusters, sparse_keep)
            fused_full = mapmos_full.copy()
            fused_full[in_range] = fused_point
            added += int((fused_full & ~mapmos_full).sum())
            removed += int((mapmos_full & ~fused_full).sum())

            fused_raw = mapmos_raw.copy()
            upper = fused_raw[fused_full] & np.uint32(0xFFFF0000)
            fused_raw[fused_full] = upper | np.uint32(args.output_moving_id)
            # MapMOS uses semantic 9 for static; preserve every existing static
            # value instead of inventing labels for vetoed moving points.
            vetoed = mapmos_full & ~fused_full
            upper = fused_raw[vetoed] & np.uint32(0xFFFF0000)
            fused_raw[vetoed] = upper | np.uint32(9)
            destination = output_dir / f"{frame:06d}.label"
            atomic_u32(fused_raw, destination)
            output_names.append(destination.name)
            written_points += len(fused_raw)
            if batch_index % 200 == 0:
                print(f"  frame {batch_index}/{len(loader)}", flush=True)

    actual = sorted(path.name for path in output_dir.glob("*.label"))
    if actual != sorted(output_names):
        raise RuntimeError("unexpected output label set")
    manifest = {
        "method": "frozen_mapmos_sparse_risk_hybrid",
        "sequence": args.sequence,
        "frames": len(output_names),
        "points": written_points,
        "regime": regime,
        "raw_mapmos_prevalence": prevalence,
        "prevalence_gate": PREVALENCE_GATE,
        "margin_factor": args.margin_factor,
        "veto_threshold": VETO_THRESHOLD,
        "recovery_add_threshold": add_threshold,
        "added_predictions": added,
        "removed_predictions": removed,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resolver": str(resolver_path),
        "resolver_sha256": sha256_file(resolver_path),
        "mapmos_prediction_dir": str(mapmos_dir),
        "ground_truth_read": False,
    }
    manifest_path = output_dir / "fusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n=== frozen hybrid export complete ===")
    print(f"sequence={args.sequence:02d} regime={regime}")
    print(f"output={output_dir}")
    print(f"frames={len(output_names)} points={written_points} "
          f"added={added} removed={removed}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
