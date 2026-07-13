"""v3 center-space propagation, sweeping center linkage in ONE forward pass.
Rules in/out 'v3 lost only because 0.3m linkage was wrong for 0.72m offsets'.
Same checkpoint, same gate; only clustering linkage varies.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_v3_linksweep.py \
    --config ~/sparseunet4d/configs/offset_v1.yaml \
    --ckpt ~/sparseunet4d/runs/offset_full/best.pt
"""
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader
from scipy import ndimage

TH = 0.5
LINKS = [0.2, 0.3, 0.4, 0.5]     # center-space linkage sweep (meters)
FRACS = [0.05, 0.10, 0.20, 0.30]
MAX_CLUSTER = 15000
MIN_CLUSTER = 5
MOVABLE_SEM = {1, 2, 3, 4, 5, 6, 7, 8}


def comp(pts_m, link_m):
    if len(pts_m) == 0:
        return np.zeros(0, np.int64)
    c = np.floor(pts_m / link_m).astype(np.int64); c -= c.min(0)
    grid = np.zeros(c.max(0) + 1, dtype=bool)
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    return lab[c[:, 0], c[:, 1], c[:, 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vs = d["voxel_size"]

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)

    NL, KF = len(LINKS), len(FRACS)
    tp = np.zeros((NL, KF), np.int64); fp = np.zeros((NL, KF), np.int64)
    fn = np.zeros((NL, KF), np.int64)
    b_tp = b_fp = b_fn = 0

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
            sem_pred = out["semantic_logits"].argmax(1).cpu().numpy()
            offset = out["offset_pred"].cpu().numpy()
            gt = batch["motion"].numpy(); cnp = batch["coords"].cpu().numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup]; sp = sem_pred[sup]
            xyz_m = cnp[sup][:, 1:4].astype(np.float64) * vs
            off = offset[sup]; gpos = g == 1
            base = pm >= TH
            b_tp += int((base & gpos).sum()); b_fp += int((base & ~gpos).sum())
            b_fn += int((~base & gpos).sum())

            movable = np.isin(sp, list(MOVABLE_SEM))
            mov_idx = np.where(movable)[0]
            centers = xyz_m[movable] + off[movable]
            for li, L in enumerate(LINKS):
                lab = comp(centers, L)
                preds = [base.copy() for _ in FRACS]
                for lid in np.unique(lab):
                    sel = lab == lid; n = int(sel.sum())
                    if n > MAX_CLUSTER or n < MIN_CLUSTER:
                        continue
                    idx = mov_idx[sel]; fconf = base[idx].mean()
                    for k, F in enumerate(FRACS):
                        if fconf >= F:
                            preds[k][idx] = True
                for k in range(KF):
                    pr = preds[k]
                    tp[li, k] += int((pr & gpos).sum())
                    fp[li, k] += int((pr & ~gpos).sum())
                    fn[li, k] += int((~pr & gpos).sum())
            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    b_iou = b_tp / max(b_tp + b_fp + b_fn, 1)
    print(f"\n=== v3 center-link sweep, val seq {d['val_sequences']} ===")
    print(f"baseline argmax IoU={b_iou:.4f}  | v2 best 0.6430 | oracle 0.769")
    print(f"{'link(m)':>8} {'frac':>6} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'FP':>9}")
    best = (-1, None)
    for li, L in enumerate(LINKS):
        for k, F in enumerate(FRACS):
            iou = tp[li, k] / max(tp[li, k] + fp[li, k] + fn[li, k], 1)
            prec = tp[li, k] / max(tp[li, k] + fp[li, k], 1)
            rec = tp[li, k] / max(tp[li, k] + fn[li, k], 1)
            if iou > best[0]:
                best = (iou, (L, F))
            print(f"{L:8.2f} {F:6.2f} {iou:8.4f} {prec:8.4f} {rec:8.4f} {fp[li,k]:9d}")
        print()
    print(f"best: IoU {best[0]:.4f} @ link={best[1][0]}m frac={best[1][1]}")
    print("If best <= 0.6430: center space is corrupted (offsets on unsupervised "
          "static-movable voxels are garbage) -> not a linkage problem. Verdict "
          "stands: naive offset head underperforms; needs panoptic offset "
          "supervision + decoupled head, or ship the v2 0.6430 result.")


if __name__ == "__main__":
    main()