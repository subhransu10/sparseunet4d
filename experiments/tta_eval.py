"""Test-time augmentation (TTA) for MOS — no retraining, works on any ckpt.

Each reference scan is run under the D4 symmetry group (0/90/180/270 deg
rotations x flips about z). D4 is special: it only permutes/negates the x,y
axes, so the *box* range-clip keeps exactly the same points in the same order
across views -> a scan's per-point moving-probabilities are averageable across
views by point index. Rotations that aren't multiples of 90 deg, or any scaling,
would change which points survive the box clip and break correspondence.

We fuse by mean probability per reference point, then score point-level (the
official-protocol granularity). Reports single-view baseline vs K-view TTA.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/tta_eval.py \
    --config ~/sparseunet4d/configs/residual_temporal.yaml \
    --ckpt ~/sparseunet4d/runs/residual_temporal/best.pt \
    --views 8 --threshold 0.2
"""
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader


def d4_transforms():
    """The 8 elements of D4 as 3x3 matrices (identity on z). Each preserves the
    axis-aligned box range-clip, so point membership/order is view-invariant."""
    xy = [
        [[1, 0], [0, 1]],    # identity
        [[0, -1], [1, 0]],   # rot 90
        [[-1, 0], [0, -1]],  # rot 180
        [[0, 1], [-1, 0]],   # rot 270
        [[-1, 0], [0, 1]],   # flip x
        [[1, 0], [0, -1]],   # flip y
        [[0, 1], [1, 0]],    # transpose (diag flip)
        [[0, -1], [-1, 0]],  # anti-diag flip
    ]
    mats = []
    for m in xy:
        M = np.eye(3, dtype=np.float32)
        M[:2, :2] = np.array(m, np.float32)
        mats.append(M)
    return mats


def build_loader(d, p, transform):
    ds = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"],
        residual_feats=d.get("residual_feats", True), res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"), return_point_map=True,
        fixed_transform=transform)
    return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=me_collate,
                      num_workers=4)


def per_point_prob(model, batch, dev):
    coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
    if backend() == "me":
        import MinkowskiEngine as ME
        x = ME.SparseTensor(feats, coordinates=coords)
    else:
        from sparseunet4d.models.backend import ST
        x = ST(feats, coords)
    out = model(x)
    prob = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
    rpv = batch["ref_point_voxel"].numpy()
    return prob[rpv], batch["ref_point_motion"].numpy()   # (n_ref,), (n_ref,)


def iou_at(prob, gt, th):
    m = gt != -1
    pr = (prob[m] >= th).astype(np.int64); g = gt[m]
    tp = int(((pr == 1) & (g == 1)).sum())
    fp = int(((pr == 1) & (g == 0)).sum())
    fn = int(((pr == 0) & (g == 1)).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return iou, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--views", type=int, default=8, help="1..8 D4 views")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    views = d4_transforms()[:max(1, min(args.views, 8))]
    prob_sum = None; gt_all = None; single = None
    with torch.no_grad():
        for vi, R in enumerate(views):
            loader = build_loader(d, p, R)
            chunks, gts = [], []
            for bi, batch in enumerate(loader):
                pr, g = per_point_prob(model, batch, dev)
                chunks.append(pr)
                if vi == 0:
                    gts.append(g)
                if bi % 500 == 0:
                    print(f"  view {vi} ({['id','r90','r180','r270','fx','fy','td','ta'][vi]}) "
                          f"frame {bi}/{len(loader)}", flush=True)
            v = np.concatenate(chunks)
            if vi == 0:
                prob_sum = v.copy(); gt_all = np.concatenate(gts); single = v.copy()
            else:
                if v.shape != prob_sum.shape:
                    raise RuntimeError(
                        f"view {vi} point count {v.shape} != baseline {prob_sum.shape} "
                        "-- correspondence broke (non-D4 transform?)")
                prob_sum += v
    prob_tta = prob_sum / len(views)

    print(f"\n=== TTA on val seq {d['val_sequences']}  ({len(views)} D4 views) ===")
    print(f"{'setting':>16} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
    for name, prob in [("single-view", single), (f"TTA x{len(views)}", prob_tta)]:
        iou, prec, rec = iou_at(prob, gt_all, args.threshold)
        print(f"{name:>16} {iou:8.4f} {prec:8.4f} {rec:8.4f}")
    # small threshold sweep on the fused probs (TTA can shift the best cutoff)
    print("\n  fused-prob threshold sweep:")
    best = (-1, None)
    for th in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        iou, prec, rec = iou_at(prob_tta, gt_all, th)
        if iou > best[0]:
            best = (iou, th)
        print(f"    th={th:.2f}  IoU={iou:.4f}  P={prec:.4f}  R={rec:.4f}")
    print(f"  best TTA IoU {best[0]:.4f} @ th {best[1]}")


if __name__ == "__main__":
    main()
