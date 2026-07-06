"""Diagnose the ~0.60 ceiling by LOOKING at one validation sample.

Prints hard numbers that catch the usual MOS pipeline bugs, and (if a checkpoint
is given) compares prediction vs ground truth. Also dumps a .npy of points so
you can view them in any 3D viewer. No training, no GPU needed for the audit
parts (checkpoint eval uses GPU if available).

Checks:
  1. Moving-voxel fraction in GT  -> is the class absurdly rare / empty?
  2. Are past frames actually registered (do static structures overlap across t)?
  3. Do moving objects leave a temporal 'trail' (spread across frames)?
  4. Label sanity: unique semantic ids, ignore fraction, motion/semantic agree?
  5. (with --ckpt) confusion on moving class: TP/FP/FN, precision/recall.

Usage:
  PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d SU4D_BACKEND=me python3 inspect_sample.py \
      --config ~/sparseunet4d/configs/semantickitti_base.yaml --idx 0 [--ckpt ~/runs/full/best.pt]
"""
import os, sys, argparse, yaml
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dump", default=None, help="path to save points .npy for 3D viewing")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]

    # build the VAL set exactly as training sees it (clean poses)
    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"])
    s = ds[args.idx]
    coords = s["coords"]; feats = s["feats"]; mot = s["motion"]; sem = s["semantic"]
    if hasattr(coords, "numpy"): coords = coords.numpy()
    if hasattr(mot, "numpy"): mot = mot.numpy()
    if hasattr(sem, "numpy"): sem = sem.numpy()
    t = coords[:, 3]

    print(f"=== sample idx {args.idx}  seq/ref {s['meta']} ===")
    print(f"total voxels: {len(coords)}   frames present (t): {sorted(np.unique(t).tolist())}")

    # 1. reference-frame supervision + moving fraction
    ref = t == 0
    n_ref = ref.sum()
    sup = mot[ref] != -1
    mov = (mot[ref] == 1).sum()
    print(f"\n[1] reference voxels: {n_ref}  supervised: {sup.sum()}  "
          f"moving: {mov} ({100*mov/max(n_ref,1):.2f}% of ref)")
    if mov == 0:
        print("    !! NO moving voxels in this sample's reference frame.")

    # 2. registration check: do static structures overlap across frames?
    # count spatial cells (x,y,z) shared between t=0 and t=1
    if (t == 1).any():
        c0 = set(map(tuple, coords[t == 0][:, 1:4].tolist()))
        c1 = set(map(tuple, coords[t == 1][:, 1:4].tolist()))
        overlap = len(c0 & c1) / max(len(c0), 1)
        print(f"\n[2] spatial-cell overlap t0∩t1: {100*overlap:.1f}% "
              f"(high => frames are registered; near 0 => NOT aligned)")

    # 3. moving 'trail': spatial spread of moving voxels vs static (per ref)
    # (uses reference-frame moving voxels only; sanity that motion labels localise)
    print(f"\n[3] moving-voxel spatial extent (ref): "
          f"{'n/a' if mov==0 else np.ptp(coords[ref][mot[ref]==1][:, 1:4], axis=0).tolist()}")

    # 4. label sanity
    ig = (sem[ref] == -1).mean()
    print(f"\n[4] semantic ids present (ref): {sorted(np.unique(sem[ref]).tolist())}")
    print(f"    semantic ignore fraction (ref): {100*ig:.1f}%")
    # motion vs semantic agreement: moving voxels should have valid semantic too
    if mov > 0:
        movsem = sem[ref][mot[ref] == 1]
        print(f"    semantic ids on MOVING voxels: {sorted(np.unique(movsem).tolist())}")

    # 5. prediction confusion (optional)
    if args.ckpt:
        import torch
        from sparseunet4d.models.backend import ST, backend
        from sparseunet4d.models.model import SparseUNet4D
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = cfg["model"]
        model = SparseUNet4D(1, d.get("num_semantic", 20), base=m.get("base", 32),
            n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
            use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
        ck = torch.load(args.ckpt, map_location=dev)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        cc = torch.from_numpy(np.concatenate([np.zeros((len(coords),1),np.int32), coords],1)) \
            if coords.shape[1] == 4 else torch.from_numpy(coords)
        # collate single sample: prepend batch col
        bcol = np.zeros((len(coords), 1), np.int32)
        C = torch.from_numpy(np.concatenate([bcol, coords], 1)).int().to(dev)
        F = torch.from_numpy(feats.numpy() if hasattr(feats,"numpy") else feats).float().to(dev)
        x = (__import__("MinkowskiEngine").SparseTensor(F, coordinates=C)
             if backend()=="me" else ST(F, C))
        with torch.no_grad():
            out = model(x)
        pred = out["motion_logits"].argmax(1).cpu().numpy()
        gt = mot
        m_ref = (gt != -1)
        pr, g = pred[m_ref], gt[m_ref]
        tp = int(((pr==1)&(g==1)).sum()); fp=int(((pr==1)&(g==0)).sum()); fn=int(((pr==0)&(g==1)).sum())
        iou = tp/max(tp+fp+fn,1)
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        print(f"\n[5] moving-class on this sample: TP={tp} FP={fp} FN={fn}  "
              f"IoU={iou:.3f} precision={prec:.3f} recall={rec:.3f}")
        if fp > 3*tp: print("    !! many FALSE POSITIVES -> over-predicting motion (drift/registration?)")
        if fn > 3*tp: print("    !! many FALSE NEGATIVES -> missing moving objects (labels/features weak?)")

    if args.dump:
        out = np.concatenate([coords[ref][:, 1:4], mot[ref][:, None]], 1)
        np.save(args.dump, out)
        print(f"\nsaved reference points+motion to {args.dump} (cols: x,y,z,moving)")


if __name__ == "__main__":
    main()