"""Sweep the moving-probability threshold at POINT level in a single pass.

Every point-level number so far used the threshold tuned on the VOXEL meter
during training. The point-level optimum generally differs. This runs the model
once over val-08, caches per-point probabilities + GT, then evaluates the whole
threshold grid -- so you get the true operating point for free (no retraining,
one forward pass instead of one per threshold).

Usage:
  SU4D_BACKEND=me PYTHONPATH=$HOME/MinkowskiEngine:$HOME/sparseunet4d \
  python sweep_threshold_pointlevel.py --config configs/dual_v4.yaml \
    --ckpt runs/dual_v4/best.pt
"""
from __future__ import annotations
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                             0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; m = cfg["model"]; p = cfg.get("pose", {"seed": 0})
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p.get("seed", 0),
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), return_point_map=True,
        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
    from scripts.train import build_model
    model = build_model(m, in_ch, d.get("num_semantic", 20)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    # accumulate per-threshold counts in ONE pass (probs are not stored wholesale)
    grid = np.array(sorted(args.grid), dtype=np.float32)
    tp = np.zeros(len(grid), np.int64)
    fp = np.zeros(len(grid), np.int64)
    fn = np.zeros(len(grid), np.int64)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
            if backend() == "me":
                import MinkowskiEngine as ME
                x = ME.SparseTensor(feats, coordinates=coords)
            else:
                from sparseunet4d.models.backend import ST
                x = ST(feats, coords)
            out = model(x)
            prob = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
            pm = prob[batch["ref_point_voxel"].numpy()]
            g = batch["ref_point_motion"].numpy()
            msk = g != -1
            pm, gm = pm[msk], (g[msk] == 1)
            for k, th in enumerate(grid):
                pr = pm >= th
                tp[k] += int((pr & gm).sum())
                fp[k] += int((pr & ~gm).sum())
                fn[k] += int((~pr & gm).sum())
            if bi % 200 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)

    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    best = int(np.argmax(iou))
    print("\n=== point-level threshold sweep (val-08) ===")
    print(f"{'th':>6} {'IoU':>8} {'prec':>8} {'rec':>8}")
    for k, th in enumerate(grid):
        star = "  <== best" if k == best else ""
        print(f"{th:6.2f} {iou[k]:8.4f} {prec[k]:8.4f} {rec[k]:8.4f}{star}")
    print(f"\nBEST: IoU {iou[best]:.4f} @ threshold {grid[best]:.2f} "
          f"(P={prec[best]:.4f} R={rec[best]:.4f})")


if __name__ == "__main__":
    main()
