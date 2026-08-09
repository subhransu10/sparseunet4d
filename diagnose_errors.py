"""Where are the remaining errors? Point-level FN/FP structure on val-08.

Answers the three questions that decide the next architecture move:
  1. RANGE   - are misses far-field (sensor sparsity) or near-field (model)?
  2. CLASS   - which mover types do we miss (car vs person vs cyclist vs truck)?
  3. OBJECT  - are misses whole instances (object-level reasoning still missing)
               or thin margins on detected instances (calibration/boundary)?
               Includes the INSTANCE-ORACLE ceiling: IoU if every instance with
               >=X% of its points detected were completed perfectly. That number
               is the headroom left for any object/cluster-level method.

Usage:
  SU4D_BACKEND=me PYTHONPATH=$HOME/MinkowskiEngine:$HOME/sparseunet4d \
  python diagnose_errors.py --config configs/dual_v4.yaml \
      --ckpt runs/dual_v4/best.pt --threshold 0.5
"""
from __future__ import annotations
import os, sys, argparse, yaml
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import MOVING_IDS, split_label
from sparseunet4d.datasets.semantickitti import _read_scan, _read_label
from sparseunet4d.models.backend import backend
from torch.utils.data import DataLoader

# raw SemanticKITTI moving ids -> readable name.
# NOTE the ordering: the MOVING ids put bicyclist BEFORE person (253/254),
# the opposite of the static ids (30 person / 31 bicyclist). Easy to get wrong.
MOV_NAME = {252: "car", 253: "bicyclist", 254: "person", 255: "motorcyclist",
            256: "on-rails", 257: "bus", 258: "truck", 259: "other-vehicle"}
RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 52)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; m = cfg["model"]; p = cfg.get("pose", {"seed": 0})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = d["point_range"]

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p.get("seed", 0),
        R, residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), return_point_map=True,
        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=me_collate,
                        num_workers=4)

    _k = (d.get("n_frames", 4) - 1) * (2 if d.get("residual_validity", False) else 1)
    in_ch = 1 + _k if d.get("residual_feats", True) else 1
    from scripts.train import build_model
    model = build_model(m, in_ch, d.get("num_semantic", 20)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    rng_tp = np.zeros(len(RANGE_BINS)); rng_fn = np.zeros(len(RANGE_BINS))
    rng_fp = np.zeros(len(RANGE_BINS))
    cls_tp = defaultdict(int); cls_fn = defaultdict(int)
    # per-instance detected fraction
    inst_frac = []                 # (n_points, detected_fraction)
    TP = FP = FN = 0

    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            seq, ref = ds.index[bi]
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

            # raw labels / geometry for the SAME reference points (t=0: no transform)
            seq_dir = os.path.join(d["root"], f"{seq:02d}")
            scan = _read_scan(os.path.join(seq_dir, "velodyne", f"{ref:06d}.bin"))
            xyz = scan[:, :3]
            keep = np.all(np.abs(xyz) < R, axis=1)
            sem_raw, inst_raw = split_label(
                _read_label(os.path.join(seq_dir, "labels", f"{ref:06d}.label")))
            if len(sem_raw) != len(xyz) or keep.sum() != len(g):
                continue                       # skip any misaligned frame
            sem_k, inst_k, xyz_k = sem_raw[keep], inst_raw[keep], xyz[keep]
            rad = np.linalg.norm(xyz_k, axis=1)

            pred = pm >= args.threshold
            gt = g == 1
            valid = g != -1
            TP += int((pred & gt & valid).sum())
            FP += int((pred & ~gt & valid).sum())
            FN += int((~pred & gt & valid).sum())

            for k, (lo, hi) in enumerate(RANGE_BINS):
                b = (rad >= lo) & (rad < hi) & valid
                rng_tp[k] += int((pred & gt & b).sum())
                rng_fn[k] += int((~pred & gt & b).sum())
                rng_fp[k] += int((pred & ~gt & b).sum())

            for rid in np.unique(sem_k[gt & valid]):
                sel = (sem_k == rid) & gt & valid
                cls_tp[int(rid)] += int((pred & sel).sum())
                cls_fn[int(rid)] += int((~pred & sel).sum())

            mov = gt & valid
            for iid in np.unique(inst_k[mov]):
                sel = mov & (inst_k == iid)
                n = int(sel.sum())
                if n >= 5:
                    inst_frac.append((n, float(pred[sel].mean())))

            n_done += 1
            if bi % 200 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)
            if args.max_frames and n_done >= args.max_frames:
                break

    iou = TP / max(TP + FP + FN, 1)
    print(f"\n==== overall (th={args.threshold}, {n_done} frames) ====")
    print(f"IoU {iou:.4f}  P {TP/max(TP+FP,1):.4f}  R {TP/max(TP+FN,1):.4f}"
          f"  TP {TP} FP {FP} FN {FN}")

    print("\n==== by RANGE (moving points) ====")
    print(f"{'bin(m)':>10} {'recall':>8} {'prec':>8} {'GTmov':>10} {'FN':>10} {'FP':>10}")
    for k, (lo, hi) in enumerate(RANGE_BINS):
        gtm = rng_tp[k] + rng_fn[k]
        rec = rng_tp[k] / max(gtm, 1); pre = rng_tp[k] / max(rng_tp[k] + rng_fp[k], 1)
        print(f"{lo:4d}-{hi:<5d} {rec:8.3f} {pre:8.3f} {int(gtm):10d} "
              f"{int(rng_fn[k]):10d} {int(rng_fp[k]):10d}")

    print("\n==== by MOVER CLASS ====")
    print(f"{'class':>14} {'recall':>8} {'GTpts':>10} {'FN':>10}")
    for rid in sorted(cls_tp.keys() | cls_fn.keys(),
                      key=lambda r: -(cls_tp[r] + cls_fn[r])):
        tot = cls_tp[rid] + cls_fn[rid]
        print(f"{MOV_NAME.get(rid, rid):>14} {cls_tp[rid]/max(tot,1):8.3f} "
              f"{tot:10d} {cls_fn[rid]:10d}")

    print("\n==== by INSTANCE (are misses whole objects?) ====")
    if inst_frac:
        fr = np.array([f for _, f in inst_frac])
        npts = np.array([n for n, _ in inst_frac])
        buckets = [(0.0, 0.1, "missed entirely"), (0.1, 0.5, "mostly missed"),
                   (0.5, 0.9, "partially found"), (0.9, 1.01, "fully found")]
        print(f"{'bucket':>18} {'#inst':>7} {'%inst':>7} {'%mov pts':>9}")
        for lo, hi, name in buckets:
            s = (fr >= lo) & (fr < hi)
            print(f"{name:>18} {int(s.sum()):7d} {100*s.mean():7.1f} "
                  f"{100*npts[s].sum()/max(npts.sum(),1):9.1f}")
        # instance-oracle: complete every instance already >=X% detected
        print("\n  instance-oracle ceiling (complete partially-detected instances):")
        for x in (0.1, 0.25, 0.5):
            recovered = int(npts[(fr >= x)].sum() * 1.0)
            got = int((npts * fr).sum())
            extra = max(recovered - got, 0)
            o_tp = TP + extra; o_fn = max(FN - extra, 0)
            print(f"    detected>= {x:>4.0%}: IoU -> "
                  f"{o_tp/max(o_tp+FP+o_fn,1):.4f}")
    print("\nInterpretation: far-field-dominated FN => sensor/density limit "
          "(more capacity won't fix it). Whole-instance misses with a high "
          "oracle ceiling => object-level reasoning still has headroom. "
          "Uniform thin margins => calibration/boundary, not structure.")


if __name__ == "__main__":
    main()
