"""IoU-vs-drift sweep — the headline robustness result.

Loads ONE trained checkpoint, then re-runs validation at each pose-noise level
(no retraining). For every (rot_std, trans_std) pair it rebuilds the val set
with pose_mode='drift', evaluates, and records moving-IoU + semantic-mIoU.
Outputs a CSV table and a PNG curve.

Usage:
  SU4D_BACKEND=me python scripts/eval_robustness.py \
      --config configs/semantickitti_base.yaml --ckpt runs/best.pt
"""
from __future__ import annotations
import os, sys, csv, argparse, yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from scripts.train import validate


def build_val_loader(cfg, rot_std, trans_std):
    d = cfg["dataset"]
    ds = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], pose_mode="drift",
        rot_std_deg=rot_std, trans_std_m=trans_std,
        pose_seed=cfg["pose"].get("seed", 0), point_range=d["point_range"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                      shuffle=False, collate_fn=me_collate, num_workers=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="robustness")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    num_sem = cfg["dataset"].get("num_semantic", 20)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SparseUNet4D(
        in_ch=1, num_semantic=num_sem,
        use_se=cfg["model"].get("use_se", True),
        use_ego_decouple=cfg["model"].get("use_ego_decouple", True)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rot_sweep = cfg["robustness"]["rot_std_deg_sweep"]
    trans_sweep = cfg["robustness"]["trans_std_m_sweep"]
    # paired sweep (level 0 = clean, increasing drift); zip rot & trans
    rows = []
    for rot, trans in zip(rot_sweep, trans_sweep):
        loader = build_val_loader(cfg, rot, trans)
        m = validate(model, loader, cfg, device, num_sem)
        rows.append((rot, trans, m["moving_iou"], m["semantic_miou"]))
        print(f"rot={rot:>4}deg trans={trans:>4}m  "
              f"moving_IoU={m['moving_iou']:.4f}  sem_mIoU={m['semantic_miou']:.4f}")

    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rot_std_deg", "trans_std_m", "moving_iou", "semantic_miou"])
        w.writerows(rows)
    print(f"saved {args.out}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = list(range(len(rows)))
        labels = [f"{r:.2g}/{t:.2g}" for r, t, _, _ in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(x, [r[2] for r in rows], "-o", label="moving IoU")
        plt.plot(x, [r[3] for r in rows], "-s", label="semantic mIoU")
        plt.xticks(x, labels); plt.xlabel("drift (rot deg / trans m)")
        plt.ylabel("IoU"); plt.title("Robustness to odometry drift")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{args.out}.png", dpi=150)
        print(f"saved {args.out}.png")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
