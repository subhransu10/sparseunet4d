"""Build a bank of moving instances WITH their multi-frame trajectories.

For each sampled reference frame f in the train sequences, every moving
instance (raw sem 252-259, temporally consistent instance id) is extracted at
f and at each strided past offset o (f-o), with the past points registered
into f's sensor frame via GT poses. The result is the object's real geometry
at each stack position AND its real ego-compensated displacement — exactly
what the residual channels respond to.

At train time (see SemanticKITTI4D inject_* params) an instance is pasted into
a scene by one rigid transform (yaw + xy translation) applied to ALL of its
frames, so the trajectory stays physically consistent and the spherical
residuals computed afterwards are genuine.

Usage (on the machine with the dataset):
  python3 build_instance_bank.py \
      --root "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences" \
      --seqs 0 1 2 3 4 5 6 7 9 10 --offsets 1 2 4 8 \
      --out mover_bank.npy
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparseunet4d.datasets.poses import GTPoseProvider
from sparseunet4d.datasets.label_map import MOVING_IDS, split_label
from sparseunet4d.datasets.semantickitti import _read_scan, _read_label, _transform


def extract_instance(root, seq, provider, f, sem_id, inst_id, offsets,
                     max_pts, min_pts_past):
    """Return {offset: (N,4) float32 [x,y,z,rem] in frame-f sensor coords} or None."""
    seq_dir = os.path.join(root, f"{seq:02d}")
    frames = {}
    for o in [0] + list(offsets):
        g = f - o
        if g < 0:
            continue
        bin_p = os.path.join(seq_dir, "velodyne", f"{g:06d}.bin")
        lab_p = os.path.join(seq_dir, "labels", f"{g:06d}.label")
        if not (os.path.exists(bin_p) and os.path.exists(lab_p)):
            continue
        scan = _read_scan(bin_p)
        sem_raw, inst_raw = split_label(_read_label(lab_p))
        m = (sem_raw == sem_id) & (inst_raw == inst_id)
        if m.sum() < (min_pts_past if o > 0 else min_pts_past * 3):
            continue
        pts = scan[m]                                   # (n,4)
        if o > 0:
            T = provider.relative(g, f)
            pts = np.concatenate(
                [_transform(pts[:, :3], T).astype(np.float32), pts[:, 3:4]], 1)
        if len(pts) > max_pts:
            sel = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
            pts = pts[sel]
        frames[o] = pts.astype(np.float32)
    # need the reference view and at least 2 past views to be a useful trajectory
    if 0 not in frames or sum(1 for o in frames if o > 0) < 2:
        return None
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seqs", type=int, nargs="+", required=True)
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="mover_bank.npy")
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--min-pts", type=int, default=40,
                    help="min points per PAST view (ref needs 3x this)")
    ap.add_argument("--max-pts", type=int, default=2048)
    ap.add_argument("--per-seq", type=int, default=150)
    ap.add_argument("--range-m", type=float, default=45.0,
                    help="keep instances whose ref centroid is within this range")
    args = ap.parse_args()

    bank = []
    for seq in args.seqs:
        seq_dir = os.path.join(args.root, f"{seq:02d}")
        provider = GTPoseProvider(seq_dir)
        n = len([x for x in os.listdir(os.path.join(seq_dir, "velodyne"))
                 if x.endswith(".bin")])
        found = 0
        for f in range(max(args.offsets), n, args.frame_stride):
            if found >= args.per_seq:
                break
            lab_p = os.path.join(seq_dir, "labels", f"{f:06d}.label")
            if not os.path.exists(lab_p):
                continue
            sem_raw, inst_raw = split_label(_read_label(lab_p))
            moving = np.isin(sem_raw, list(MOVING_IDS)) & (inst_raw > 0)
            if not moving.any():
                continue
            keys = np.unique(sem_raw[moving].astype(np.int64) * 100000
                             + inst_raw[moving].astype(np.int64))
            for key in keys:
                if found >= args.per_seq:
                    break
                sem_id, inst_id = int(key // 100000), int(key % 100000)
                inst = extract_instance(args.root, seq, provider, f, sem_id,
                                        inst_id, args.offsets,
                                        args.max_pts, args.min_pts)
                if inst is None:
                    continue
                c = inst[0][:, :3].mean(0)
                if np.linalg.norm(c[:2]) > args.range_m:
                    continue
                # displacement over the widest available offset: keep genuine movers
                widest = max(o for o in inst if o > 0)
                disp = np.linalg.norm(inst[0][:, :3].mean(0)
                                      - inst[widest][:, :3].mean(0))
                bank.append({"seq": seq, "frame": f, "sem_raw": sem_id,
                             "frames": inst, "disp_m": float(disp)})
                found += 1
        print(f"seq {seq:02d}: {found} instances (bank={len(bank)})", flush=True)

    disps = np.array([b["disp_m"] for b in bank])
    print(f"\nbank: {len(bank)} instances | displacement over widest offset: "
          f"median {np.median(disps):.2f}m  p10 {np.percentile(disps,10):.2f}m  "
          f"p90 {np.percentile(disps,90):.2f}m")
    np.save(args.out, np.array(bank, dtype=object), allow_pickle=True)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
