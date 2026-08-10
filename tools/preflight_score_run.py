"""Fail-fast preflight for dual_v11+ score experiments (no training)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", help="optional warm-start checkpoint to load strictly")
    ap.add_argument("--sample", type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, p = cfg["dataset"], cfg.get("pose", {"mode": "gt", "seed": 0})

    from sparseunet4d.datasets import SemanticKITTI4D
    from scripts.train import build_model

    ds = SemanticKITTI4D(
        d["root"], d["train_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], p.get("mode", "gt"), p.get("rot_std_deg", 0.0),
        p.get("trans_std_m", 0.0), p.get("seed", 0), d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"),
        augment=False, inject_bank=d.get("inject_bank"), inject_prob=0.0,
        inject_max_n=d.get("inject_max_n", 2),
        all_frame_labels=d.get("all_frame_labels", False),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        inject_class_boost=d.get("inject_class_boost"),
        inject_all_frame_labels=d.get("inject_all_frame_labels", False),
    )
    sample = ds[min(args.sample, len(ds) - 1)]
    k = (d["n_frames"] - 1) * (2 if d.get("residual_validity", False) else 1)
    expected = 1 + k if d.get("residual_feats", True) else 1
    assert sample["feats"].shape[1] == expected, (
        sample["feats"].shape, expected)
    assert np.isfinite(sample["feats"]).all()

    t = sample["coords"][:, 3]
    residual = sample["feats"][:, 1:]
    ref_nz = float((np.abs(residual[t == 0]).max(1) > 1e-3).mean())
    ctx = t > 0
    ctx_nz = float((np.abs(residual[ctx]).max(1) > 1e-3).mean()) if ctx.any() else 0.0
    instances = np.unique(sample["motion_instance"])
    n_instances = int((instances >= 0).sum())
    print(f"sample rows={len(t):,} feat_width={expected} "
          f"ref_residual_nonzero={ref_nz:.3f} context_nonzero={ctx_nz:.3f} "
          f"moving_instances={n_instances}")
    if d.get("residual_all_frames", False):
        assert ctx_nz > 0, "dense residual config produced zero context features"

    if d.get("inject_bank"):
        bank = np.load(d["inject_bank"], allow_pickle=True)
        ids, counts = np.unique([int(x["sem_raw"]) for x in bank],
                                return_counts=True)
        hist = dict(zip(ids.tolist(), counts.tolist()))
        print(f"mover_bank={len(bank)} class_hist={hist}")
        for raw_id in (d.get("inject_class_boost") or {}):
            assert int(raw_id) in hist, f"boosted raw id {raw_id} absent from bank"

    model = build_model(cfg.get("model", {}), expected,
                        d.get("num_semantic", 20))
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
        print(f"checkpoint strict-load passed: {args.ckpt}")
    print("preflight passed")


if __name__ == "__main__":
    main()
