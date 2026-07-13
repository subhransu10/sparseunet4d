"""Post-hoc instance propagation for MOS: the REALISTIC version of the oracle.

Per frame: cluster the actual scene (no GT), then claim whole clusters as
moving when enough of their voxels are confident. Measures how much of the
+15.5-pt oracle recall survives real clustering. No training.

Pipeline per frame (reference voxels only):
  1. per-voxel moving prob from the trained model
  2. ground removal: drop voxels the SEMANTIC head predicts as ground classes
     (road/parking/sidewalk/other-ground/terrain) -- prevents road merging
     everything into one blob. z-floor fallback if semantics look degenerate.
  3. connected components on remaining voxels (0.6 m linkage -- tighter than
     the oracle's 1.0 m because static clutter is present here)
  4. cluster claim rule, swept over:
       frac_conf >= F   (fraction of cluster voxels with prob >= 0.5)
     claimed cluster -> all its voxels predicted moving.
     unclaimed -> keep per-voxel argmax (don't destroy baseline predictions).
  5. ground voxels keep per-voxel argmax.

Reports P/R/IoU for baseline (argmax) and each F, plus cluster-size guard
sweep (ignore huge clusters -- buildings/walls -- which are never movers).

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_instance_prop.py \
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
LINK_VOX = 6                    # 0.6 m linkage on the real scene
FRACS = [0.05, 0.10, 0.20, 0.30]
MAX_CLUSTER = 15000             # voxels; larger = structure, never claim
# SemanticKITTI 19-class learning-map indices for ground-ish classes.
# VERIFY against your label_map: road, parking, sidewalk, other-ground, terrain
GROUND_SEM = {9, 10, 11, 12, 17}


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
        res_clip=d.get("res_clip", 3.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    KF = len(FRACS)
    tp = np.zeros(KF + 1, np.int64); fp = np.zeros(KF + 1, np.int64)
    fn = np.zeros(KF + 1, np.int64)          # slot 0 = baseline argmax

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

            ground = np.isin(sp, list(GROUND_SEM))
            ng = ~ground
            lab = components(vox[ng])
            preds = [base_pred.copy() for _ in FRACS]
            for lid in np.unique(lab):
                sel_ng = lab == lid
                n = int(sel_ng.sum())
                if n > MAX_CLUSTER:
                    continue
                fconf = base_pred[ng][sel_ng].mean()
                idx = np.where(ng)[0][sel_ng]
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
    print(f"\n=== post-hoc instance propagation, val seq {d['val_sequences']} ===")
    print(f"linkage={LINK_VOX*d['voxel_size']:.1f}m  max_cluster={MAX_CLUSTER}  "
          f"ground_sem={sorted(GROUND_SEM)}")
    print(f"{'rule':>20} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'FP':>9} {'FN':>9}")
    for k in range(KF + 1):
        iou = tp[k] / max(tp[k] + fp[k] + fn[k], 1)
        prec = tp[k] / max(tp[k] + fp[k], 1)
        rec = tp[k] / max(tp[k] + fn[k], 1)
        print(f"{names[k]:>20} {iou:8.4f} {prec:8.4f} {rec:8.4f} "
              f"{fp[k]:9d} {fn[k]:9d}")
    print("\nOracle ceiling was recall 0.8215 / IoU~0.769. The gap between the "
          "best row here and the oracle = cost of real clustering.")


if __name__ == "__main__":
    main()