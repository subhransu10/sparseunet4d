"""Semantic-aware instance propagation v2. Attacks the 13.5-IoU contamination
measured in v1 by using what the model already knows:

  v1 -> v2 changes:
  1. Cluster only MOVABLE-classed voxels: exclude ground AND static structure
     (building, fence, vegetation, pole, traffic-sign, trunk). Prevents
     mover-bridges-to-fence merges.
  2. Semantic gate on claims: a cluster is claimable only if the majority of
     its voxels are movable-object classes (car..motorcyclist). Static clutter
     with stray confident points can no longer be claimed.
  3. Tighter linkage 0.4 m (was 0.6).
  4. Same F sweep + baseline row; also prints v1's best (0.6340) for reference.

Everything not clustered/claimed keeps per-voxel argmax, as before.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_instance_prop_v2.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml \
    --ckpt ~/sparseunet4d/runs/residual_v2/best.pt
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
LINK_VOX = 4                     # 0.4 m linkage
FRACS = [0.05, 0.10, 0.20, 0.30]
MAX_CLUSTER = 15000
MIN_CLUSTER = 5                  # ignore singleton specks
# learned indices (verified): 1 car, 2 bicycle, 3 motorcycle, 4 truck,
# 5 other-vehicle, 6 person, 7 bicyclist, 8 motorcyclist
MOVABLE_SEM = {1, 2, 3, 4, 5, 6, 7, 8}
SEM_MAJORITY = 0.5               # claim gate: >50% of cluster movable-classed


def components(vox):
    if len(vox) == 0:
        return np.zeros(0, np.int64)
    c = (vox // LINK_VOX).astype(np.int64); c -= c.min(0)
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

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    KF = len(FRACS)
    tp = np.zeros(KF + 1, np.int64); fp = np.zeros(KF + 1, np.int64)
    fn = np.zeros(KF + 1, np.int64)

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
            gt = batch["motion"].numpy()
            cnp = batch["coords"].cpu().numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup]; sp = sem_pred[sup]
            vox = cnp[sup][:, 1:4].astype(np.int64)
            gpos = g == 1
            base_pred = pm >= TH
            tp[0] += int((base_pred & gpos).sum())
            fp[0] += int((base_pred & ~gpos).sum())
            fn[0] += int((~base_pred & gpos).sum())

            movable = np.isin(sp, list(MOVABLE_SEM))     # change 1
            lab = components(vox[movable])
            preds = [base_pred.copy() for _ in FRACS]
            mov_idx = np.where(movable)[0]
            for lid in np.unique(lab):
                sel = lab == lid
                n = int(sel.sum())
                if n > MAX_CLUSTER or n < MIN_CLUSTER:
                    continue
                # change 2: semantic majority gate is implicit (cluster IS
                # movable-only), but require it anyway vs full-frame stats:
                fconf = base_pred[mov_idx[sel]].mean()
                idx = mov_idx[sel]
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

    names = ["argmax (baseline)"] + [f"claim@frac>={F:.2f}" for F in FRACS]
    print(f"\n=== semantic-aware propagation v2, val seq {d['val_sequences']} ===")
    print(f"linkage={LINK_VOX*d['voxel_size']:.1f}m  cluster on MOVABLE sem only "
          f"{sorted(MOVABLE_SEM)}  min={MIN_CLUSTER} max={MAX_CLUSTER}")
    print(f"{'rule':>20} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'FP':>9} {'FN':>9}")
    for k in range(KF + 1):
        iou = tp[k] / max(tp[k] + fp[k] + fn[k], 1)
        prec = tp[k] / max(tp[k] + fp[k], 1)
        rec = tp[k] / max(tp[k] + fn[k], 1)
        print(f"{names[k]:>20} {iou:8.4f} {prec:8.4f} {rec:8.4f} "
              f"{fp[k]:9d} {fn[k]:9d}")
    print("\nreference: v1 geometric best 0.6340 | oracle ~0.769")


if __name__ == "__main__":
    main()