"""Generate CONSISTENT random-walk drifted trajectories for a sequence.

Model: odometry error compounds. Take the GT relative motion between
consecutive frames, corrupt each with a small SE(3) increment, recompose:
    T'[0] = T[0];   T'[i] = T'[i-1] @ (R_i @ exp(xi_i)),  xi_i ~ N(0, std)
This yields an actual drifted trajectory (unlike per-pair jitter), written as
KITTI-format poses files in BOTH frames:
  poses_velodyne.txt  -> our eval (eval_posefile.py, frame=velodyne)
  poses_camera.txt    -> external baselines that read KITTI poses.txt (4DMOS)
Because both models consume the SAME files, ours and the baseline are compared
under IDENTICAL corrupted poses — no "different noise" confound.

--make-4dmos-tree also builds, per drift level, a symlinked KITTI root:
  <tree>/<level>/sequences/<seq>/{velodyne,labels}   (symlinks, no copying)
  <tree>/<level>/sequences/<seq>/calib.txt           (copy)
  <tree>/<level>/sequences/<seq>/poses.txt           (drifted, camera frame)
Point 4DMOS's data root at <tree>/<level> and run its predict per level.

Usage:
  python3 make_drifted_poses.py \
      --seq-dir "<root>/sequences/08" \
      --levels 0.25:0.05 0.5:0.1 1.0:0.2 2.0:0.4 --seed 0 \
      --out-dir drifted_poses_08 [--make-4dmos-tree drift_kitti_08]
"""
from __future__ import annotations
import os, sys, argparse, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparseunet4d.datasets.poses import (GTPoseProvider, _se3_exp,
                                         _read_calib_Tr)


def walk_drift(poses_velo, rot_std_deg, trans_std_m, seed):
    """Corrupt consecutive relative motions, recompose. Returns (F,4,4)."""
    rng = np.random.default_rng(seed)
    rot_std = np.deg2rad(rot_std_deg)
    out = np.empty_like(poses_velo)
    out[0] = poses_velo[0]
    for i in range(1, len(poses_velo)):
        rel = np.linalg.inv(poses_velo[i - 1]) @ poses_velo[i]
        xi = np.concatenate([rng.normal(0.0, rot_std, 3),
                             rng.normal(0.0, trans_std_m, 3)])
        out[i] = out[i - 1] @ (rel @ _se3_exp(xi))
    return out


def window_rpe(gt, est, offsets=(1, 2, 4, 8)):
    """Relative-pose error of est vs gt over the model's temporal window."""
    stats = {}
    for o in offsets:
        errs = []
        for i in range(o, len(gt), max(1, len(gt) // 500)):
            rg = np.linalg.inv(gt[i]) @ gt[i - o]
            re = np.linalg.inv(est[i]) @ est[i - o]
            d = np.linalg.inv(rg) @ re
            errs.append(np.linalg.norm(d[:3, 3]))
        errs = np.array(errs)
        stats[o] = (float(errs.mean()), float(np.percentile(errs, 95)))
    return stats


def write_kitti_poses(path, poses):
    with open(path, "w") as f:
        for T in poses:
            f.write(" ".join(f"{v:.9e}" for v in T[:3, :4].reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help=".../sequences/08")
    ap.add_argument("--levels", nargs="+", default=
                    ["0.25:0.05", "0.5:0.1", "1.0:0.2", "2.0:0.4"],
                    help="rotStdDeg:transStdM per consecutive frame step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="drifted_poses")
    ap.add_argument("--make-4dmos-tree", default=None,
                    help="root for per-level symlinked KITTI trees")
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.seq_dir))
    gt = GTPoseProvider(args.seq_dir)             # velodyne frame
    Tr = _read_calib_Tr(os.path.join(args.seq_dir, "calib.txt"))
    Tr_inv = np.linalg.inv(Tr)
    os.makedirs(args.out_dir, exist_ok=True)

    for lv in args.levels:
        r, t = (float(x) for x in lv.split(":"))
        name = f"drift_r{r}_t{t}"
        drift_velo = walk_drift(gt.poses, r, t, args.seed)
        rpe = window_rpe(gt.poses, drift_velo)
        print(f"{name}: window RPE (m) " + "  ".join(
            f"off{o}: {m:.3f} (p95 {p:.3f})" for o, (m, p) in rpe.items()),
            flush=True)
        lv_dir = os.path.join(args.out_dir, name)
        os.makedirs(lv_dir, exist_ok=True)
        write_kitti_poses(os.path.join(lv_dir, "poses_velodyne.txt"), drift_velo)
        # camera frame for KITTI-convention consumers: T_cam = Tr @ T_velo @ Tr^-1
        drift_cam = np.einsum("ij,fjk,kl->fil", Tr, drift_velo, Tr_inv)
        write_kitti_poses(os.path.join(lv_dir, "poses_camera.txt"), drift_cam)

        if args.make_4dmos_tree:
            sdir = os.path.join(args.make_4dmos_tree, name, "sequences", seq)
            os.makedirs(sdir, exist_ok=True)
            for sub in ("velodyne", "labels"):
                dst = os.path.join(sdir, sub)
                if not os.path.islink(dst) and not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(args.seq_dir, sub)),
                               dst)
            shutil.copy(os.path.join(args.seq_dir, "calib.txt"),
                        os.path.join(sdir, "calib.txt"))
            shutil.copy(os.path.join(lv_dir, "poses_camera.txt"),
                        os.path.join(sdir, "poses.txt"))
    print(f"\nwrote {args.out_dir}/" +
          (f" and 4DMOS trees under {args.make_4dmos_tree}/"
           if args.make_4dmos_tree else ""))


if __name__ == "__main__":
    main()
