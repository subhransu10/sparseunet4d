"""Instance propagation v3: cluster in LEARNED CENTER space (offset head).

v2 clustered raw xyz -> best 0.6430 (adjacent parked/moving cars merge).
v3 shifts each voxel by its predicted offset (voxel -> its instance center),
then clusters. Two instances that touch in xyz separate in center space because
their voxels vote to different centers. This is the milestone-1 test of learned
grouping: does it beat 0.6430 and how much of the 0.769 oracle it banks.

Only we changed vs v2: cluster coords = (xyz_m + offset_pred), tight linkage.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_instance_prop_v3.py \
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
CENTER_LINK_M = 0.3              # linkage in center space (centers concentrate)
FRACS = [0.05, 0.10, 0.20, 0.30]
MAX_CLUSTER = 15000
MIN_CLUSTER = 5
MOVABLE_SEM = {1, 2, 3, 4, 5, 6, 7, 8}   # car..motorcyclist (verified indices)


def components_m(pts_m, link_m):
    """Connected components on points given in METERS, linkage in meters."""
    if len(pts_m) == 0:
        return np.zeros(0, np.int64)
    c = np.floor(pts_m / link_m).astype(np.int64)
    c -= c.min(0)
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
        d["point_range"],
        residual_feats=d.get("residual_feats", True),
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
    assert any(k.startswith("offset_head") for k in state), \
        "checkpoint has no offset_head -- was it trained with offset_weight>0?"
    model.load_state_dict(state)

    KF = len(FRACS)
    tp = np.zeros(KF + 1, np.int64); fp = np.zeros(KF + 1, np.int64)
    fn = np.zeros(KF + 1, np.int64)
    off_mag_sum = 0.0; off_n = 0

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
            offset = out["offset_pred"].cpu().numpy()          # (N,3) meters
            gt = batch["motion"].numpy()
            cnp = batch["coords"].cpu().numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup]; sp = sem_pred[sup]
            xyz_m = cnp[sup][:, 1:4].astype(np.float64) * vs   # voxel->meters
            off = offset[sup]
            gpos = g == 1
            base_pred = pm >= TH
            tp[0] += int((base_pred & gpos).sum())
            fp[0] += int((base_pred & ~gpos).sum())
            fn[0] += int((~base_pred & gpos).sum())

            movable = np.isin(sp, list(MOVABLE_SEM))
            mov_idx = np.where(movable)[0]
            # center-space coords for movable voxels
            centers = xyz_m[movable] + off[movable]
            off_mag_sum += float(np.linalg.norm(off[movable], axis=1).sum())
            off_n += int(movable.sum())
            lab = components_m(centers, CENTER_LINK_M)

            preds = [base_pred.copy() for _ in FRACS]
            for lid in np.unique(lab):
                sel = lab == lid
                n = int(sel.sum())
                if n > MAX_CLUSTER or n < MIN_CLUSTER:
                    continue
                idx = mov_idx[sel]
                fconf = base_pred[idx].mean()
                for k, F in enumerate(FRACS):
                    if fconf >= F:
                        preds[k][idx] = True
            for k in range(KF):
                pr = preds[k]
                tp[k+1] += int((pr & gpos).sum())
                fp[k+1] += int((pr & ~gpos).sum())
                fn[k+1] += int((~pr & gpos).sum())
            if bi % 400 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    print(f"\n=== learned center-space propagation v3, val seq {d['val_sequences']} ===")
    print(f"center_link={CENTER_LINK_M}m  movable_sem={sorted(MOVABLE_SEM)}  "
          f"mean|offset_pred|={off_mag_sum/max(off_n,1):.3f}m")
    print(f"{'rule':>20} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'FP':>9} {'FN':>9}")
    names = ["argmax (baseline)"] + [f"claim@frac>={F:.2f}" for F in FRACS]
    for k in range(KF + 1):
        iou = tp[k] / max(tp[k] + fp[k] + fn[k], 1)
        prec = tp[k] / max(tp[k] + fp[k], 1)
        rec = tp[k] / max(tp[k] + fn[k], 1)
        print(f"{names[k]:>20} {iou:8.4f} {prec:8.4f} {rec:8.4f} "
              f"{fp[k]:9d} {fn[k]:9d}")
    print("\nanchors: v2 geometric best 0.6430 | oracle ~0.769")
    print("mean|offset_pred| should be ~mean GT offset (0.4m early -> lower "
          "when trained well). If ~0, offset head didn't learn -> centers = "
          "raw xyz -> expect ~v2 numbers.")


if __name__ == "__main__":
    main()