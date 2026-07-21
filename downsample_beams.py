"""Decimate KITTI 64-beam scans to N beams -> a realistic sparse sensor.

Recovers each point's laser ring from its pitch angle (HDL-64E: 64 rings over
+3..-25 deg), keeps every (64/N)-th ring. Feeding these to KISS-ICP yields a
REAL odometry pipeline degrading on a real cheaper sensor (16/32-beam), which
pushes window-RPE into the regime where drift-robustness matters -- unlike
injected pose noise.

Writes a KITTI tree (velodyne symlink-free copies of decimated .bin; labels +
calib + poses symlinked) so KISS-ICP can run on it:
  kiss_icp_pipeline "<out>" --dataloader kitti --sequence 08

Usage:
  python3 downsample_beams.py \
    --seq-dir "<root>/sequences/08" --beams 16 \
    --out "<root_16beam>/sequences/08"
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

FOV_UP, FOV_DOWN, RINGS = 3.0, -25.0, 64


def ring_index(xyz):
    r = np.linalg.norm(xyz, axis=1)
    pitch = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    frac = (FOV_UP - pitch) / (FOV_UP - FOV_DOWN)          # 0 at top .. 1 bottom
    return np.clip((frac * RINGS).astype(np.int32), 0, RINGS - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--out", required=True, help=".../sequences/08 to create")
    args = ap.parse_args()
    step = RINGS // args.beams
    velo_in = os.path.join(args.seq_dir, "velodyne")
    velo_out = os.path.join(args.out, "velodyne")
    os.makedirs(velo_out, exist_ok=True)

    files = sorted(f for f in os.listdir(velo_in) if f.endswith(".bin"))
    kept = tot = 0
    for k, f in enumerate(files):
        scan = np.fromfile(os.path.join(velo_in, f), np.float32).reshape(-1, 4)
        keep = (ring_index(scan[:, :3]) % step) == 0
        scan[keep].astype(np.float32).tofile(os.path.join(velo_out, f))
        kept += int(keep.sum()); tot += len(scan)
        if k % 500 == 0:
            print(f"  {k}/{len(files)}", flush=True)
    # symlink the rest so the tree is a valid KITTI sequence for KISS-ICP
    for sub in ("labels",):
        s, d = os.path.join(args.seq_dir, sub), os.path.join(args.out, sub)
        if not os.path.exists(d):
            os.symlink(os.path.abspath(s), d)
    for fn in ("calib.txt", "poses.txt", "times.txt"):
        s = os.path.join(args.seq_dir, fn)
        if os.path.exists(s) and not os.path.exists(os.path.join(args.out, fn)):
            os.symlink(os.path.abspath(s), os.path.join(args.out, fn))
    print(f"\n{args.beams}-beam: kept {kept/max(tot,1)*100:.1f}% of points "
          f"({kept}/{tot}) -> {velo_out}")


if __name__ == "__main__":
    main()
