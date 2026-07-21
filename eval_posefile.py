"""Evaluate a checkpoint with poses from a FILE instead of GT.

Two paper-critical uses:
  1. REAL odometry:  pip install kiss-icp;  run it over the sequence
     (kiss_icp_pipeline --dataset kitti ...); pass its poses file here.
     -> "IoU under real LiDAR-odometry error", plus the measured window-RPE
        so the noise level is quotable.
  2. Shared drifted trajectories from make_drifted_poses.py (frame=velodyne):
     the SAME files 4DMOS consumes -> identical-noise comparison.

Reports window RPE vs GT (mean/p95 per offset) and point-level moving IoU.

Usage:
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_posefile.py \
    --config ~/sparseunet4d/configs/residual_inject.yaml \
    --ckpt ~/sparseunet4d/runs/residual_inject2/best.pt \
    --pose-file drifted_poses/drift_r0.5_t0.1/poses_velodyne.txt \
    --frame velodyne --threshold 0.5
"""
from __future__ import annotations
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.poses import FilePoseProvider, GTPoseProvider
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pose-file", required=True)
    ap.add_argument("--frame", choices=["camera", "velodyne"], default="camera",
                    help="frame of the pose file. KISS-ICP KITTI output and "
                         "KITTI GT are 'camera'; make_drifted_poses "
                         "poses_velodyne.txt is 'velodyne'.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--voxel-level", action="store_true",
                    help="score dedup voxels instead of point-level")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seq = d["val_sequences"][0]
    seq_dir = os.path.join(d["root"], f"{seq:02d}")

    point_level = not args.voxel_level
    ds = SemanticKITTI4D(d["root"], [seq], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], "gt", 0.0, 0.0, 0, d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"),
        feat_rep=d.get("feat_rep", "label"), return_point_map=point_level)
    # swap in the file-based provider
    est = FilePoseProvider(args.pose_file, frame=args.frame,
                           calib_path=os.path.join(seq_dir, "calib.txt"))
    gt = ds.pose_providers[seq]
    assert len(est) == len(gt.poses), \
        f"pose file has {len(est)} rows, sequence has {len(gt.poses)} frames"
    ds.pose_providers[seq] = est

    # window RPE vs GT — the quotable noise level of this pose source
    offsets = ds.offsets
    print("window RPE vs GT (translation, m):")
    for o in offsets:
        errs = []
        for i in range(o, len(gt.poses), max(1, len(gt.poses) // 500)):
            dT = np.linalg.inv(gt.relative(i - o, i)) @ est.relative(i - o, i)
            errs.append(np.linalg.norm(dT[:3, 3]))
        errs = np.array(errs)
        print(f"  offset {o}: mean {errs.mean():.3f}  p95 "
              f"{np.percentile(errs, 95):.3f}")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)
    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    tp = fp = fn = 0
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
            if point_level:
                pm = prob[batch["ref_point_voxel"].numpy()]
                g = batch["ref_point_motion"].numpy()
            else:
                pm = prob; g = batch["motion"].numpy()
            msk = g != -1
            pr = (pm[msk] >= args.threshold); gm = g[msk] == 1
            tp += int((pr & gm).sum()); fp += int((pr & ~gm).sum())
            fn += int((~pr & gm).sum())
            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)
    iou = tp / max(tp + fp + fn, 1)
    print(f"\n=== {os.path.basename(args.pose_file)} "
          f"[{'point' if point_level else 'voxel'}-level, th={args.threshold}] ===")
    print(f"moving IoU: {iou:.4f}   P={tp/max(tp+fp,1):.4f} "
          f"R={tp/max(tp+fn,1):.4f}")


if __name__ == "__main__":
    main()
