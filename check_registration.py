"""Decisive registration check. For one multi-frame sample, compare past-frame
points to the reference cloud BEFORE vs AFTER the pose transform.

If registration works: nearest-neighbour distance from a past frame to the
reference drops from ~metres (raw, ego moved) to ~centimetres (registered).
If it does NOT drop, registration is broken -> that's the accuracy ceiling.

Usage:
  PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 check_registration.py \
      --config ~/sparseunet4d/configs/semantickitti_base.yaml --seq 8 --ref 100
"""
import os, sys, argparse, yaml
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets.poses import build_pose_provider
from sparseunet4d.datasets.semantickitti import _read_scan, _transform


def nn_median(a, b, k=4000):
    """median nearest-neighbour distance from subsample of a to b."""
    if len(a) > k:
        a = a[np.random.default_rng(0).choice(len(a), k, replace=False)]
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(b).query(a, k=1)
        return float(np.median(d))
    except ImportError:
        # brute force on a smaller subsample
        a2 = a[:800]; b2 = b[:20000]
        d = np.sqrt(((a2[:, None, :] - b2[None, :, :]) ** 2).sum(-1)).min(1)
        return float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--ref", type=int, default=100)
    ap.add_argument("--back", type=int, default=1, help="frames back to test")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dataset"]
    seq_dir = os.path.join(d["root"], f"{args.seq:02d}")
    prov = build_pose_provider(seq_dir, "gt")

    def scan(fr):
        p = os.path.join(seq_dir, "velodyne", f"{fr:06d}.bin")
        return _read_scan(p)[:, :3]

    ref_xyz = scan(args.ref)
    f = args.ref - args.back
    past_raw = scan(f)
    T = prov.relative(f, args.ref)
    past_reg = _transform(past_raw, T)

    print(f"seq {args.seq}  ref frame {args.ref}  past frame {f}")
    print(f"ref centroid:        {ref_xyz.mean(0).round(3)}")
    print(f"past raw centroid:   {past_raw.mean(0).round(3)}")
    print(f"past reg centroid:   {past_reg.mean(0).round(3)}   "
          f"(should match ref centroid if registered)")
    print(f"relative T translation: {T[:3,3].round(3)}  |t|={np.linalg.norm(T[:3,3]):.3f} m")
    d_raw = nn_median(past_raw, ref_xyz)
    d_reg = nn_median(past_reg, ref_xyz)
    print(f"\nmedian NN dist  past->ref  RAW: {d_raw:.3f} m")
    print(f"median NN dist  past->ref  REGISTERED: {d_reg:.3f} m")
    def frac_within(a, b, r, k=4000):
        if len(a) > k:
            a = a[np.random.default_rng(0).choice(len(a), k, replace=False)]
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(b).query(a, k=1)
        return float((dd < r).mean())
    vs = cfg["dataset"]["voxel_size"]
    fr_raw = frac_within(past_raw, ref_xyz, vs)
    fr_reg = frac_within(past_reg, ref_xyz, vs)
    print(f"\nfrac within 1 voxel ({vs} m)  RAW: {fr_raw:.2f}   REGISTERED: {fr_reg:.2f}")
    if fr_reg > fr_raw * 1.3:
        print("=> REGISTRATION WORKS (more points align after transform).")
    elif d_reg < d_raw * 0.6:
        print("=> REGISTRATION WORKS (distance dropped). 0% voxel overlap was a metric artifact.")
    else:
        print("=> REGISTRATION BROKEN (distance did NOT drop). This is the ceiling.")


if __name__ == "__main__":
    main()