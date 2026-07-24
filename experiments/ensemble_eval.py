"""Two-checkpoint ensemble eval (point-level, official granularity).

The temporal and aug models score ~equal IoU with different P/R balances, i.e.
partially decorrelated errors — the textbook case where averaging their
moving-probabilities beats either. One pass over val: both models run on the
same batch, per-point probs fused as w*A + (1-w)*B, thresholds swept.

Both checkpoints must share the architecture of --config's model section.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/ensemble_eval.py \
    --config ~/sparseunet4d/configs/residual_temporal.yaml \
    --ckpt-a ~/sparseunet4d/runs/residual_temporal/best.pt \
    --ckpt-b ~/sparseunet4d/runs/residual_pro_aug/best.pt
"""
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader

THRESHOLDS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]


def load_model(ckpt, d, m, dev):
    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    return model


def to_st(batch, dev):
    coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats, coordinates=coords)
    from sparseunet4d.models.backend import ST
    return ST(feats, coords)


def iou_at(prob, gt, th):
    m = gt != -1
    pr = (prob[m] >= th).astype(np.int64); g = gt[m]
    tp = int(((pr == 1) & (g == 1)).sum())
    fp = int(((pr == 1) & (g == 0)).sum())
    fn = int(((pr == 0) & (g == 1)).sum())
    return (tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1),
            tp / max(tp + fn, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--weight", type=float, default=0.5, help="weight on A")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"),
        return_point_map=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=me_collate,
                        num_workers=4)
    ma = load_model(args.ckpt_a, d, m, dev)
    mb = load_model(args.ckpt_b, d, m, dev)

    pa_l, pb_l, gt_l = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = to_st(batch, dev)
            prob_a = torch.softmax(ma(x)["motion_logits"], 1)[:, 1].cpu().numpy()
            prob_b = torch.softmax(mb(x)["motion_logits"], 1)[:, 1].cpu().numpy()
            rpv = batch["ref_point_voxel"].numpy()
            pa_l.append(prob_a[rpv]); pb_l.append(prob_b[rpv])
            gt_l.append(batch["ref_point_motion"].numpy())
            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)
    pa, pb, gt = np.concatenate(pa_l), np.concatenate(pb_l), np.concatenate(gt_l)
    pe = args.weight * pa + (1 - args.weight) * pb

    print(f"\n=== ensemble on val seq {d['val_sequences']} (w={args.weight}) ===")
    print(f"{'setting':>12} {'best IoU':>9} {'@th':>5} {'Prec':>8} {'Rec':>8}")
    for name, prob in [("A alone", pa), ("B alone", pb), ("ensemble", pe)]:
        best = max(((iou_at(prob, gt, th), th) for th in THRESHOLDS),
                   key=lambda x: x[0][0])
        (iou, prec, rec), th = best
        print(f"{name:>12} {iou:9.4f} {th:5.2f} {prec:8.4f} {rec:8.4f}")
    print("\n  ensemble threshold sweep:")
    for th in THRESHOLDS:
        iou, prec, rec = iou_at(pe, gt, th)
        print(f"    th={th:.2f}  IoU={iou:.4f}  P={prec:.4f}  R={rec:.4f}")


if __name__ == "__main__":
    main()
