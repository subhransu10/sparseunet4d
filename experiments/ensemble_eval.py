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
from torch.utils.data import DataLoader

THRESHOLDS = [x / 100 for x in range(5, 100, 5)] + [0.93]
WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def load_model(ckpt, d, m, dev):
    n_frames = d.get("n_frames", 4)
    k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)
    in_ch = 1 + k if d.get("residual_feats", True) else 1
    from scripts.train import build_model
    model = build_model(m, in_ch, d.get("num_semantic", 20)).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
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
    ap.add_argument("--weights", type=float, nargs="+", default=WEIGHTS,
                    help="weights on checkpoint A")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"),
        feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        return_point_map=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=me_collate,
                        num_workers=4)
    ma = load_model(args.ckpt_a, d, m, dev)
    mb = load_model(args.ckpt_b, d, m, dev)

    pa_l, pb_l, la_l, lb_l, gt_l = [], [], [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = to_st(batch, dev)
            logits_a = ma(x)["motion_logits"]
            logits_b = mb(x)["motion_logits"]
            prob_a = torch.softmax(logits_a, 1)[:, 1].cpu().numpy()
            prob_b = torch.softmax(logits_b, 1)[:, 1].cpu().numpy()
            margin_a = (logits_a[:, 1] - logits_a[:, 0]).cpu().numpy()
            margin_b = (logits_b[:, 1] - logits_b[:, 0]).cpu().numpy()
            rpv = batch["ref_point_voxel"].numpy()
            pa_l.append(prob_a[rpv]); pb_l.append(prob_b[rpv])
            la_l.append(margin_a[rpv]); lb_l.append(margin_b[rpv])
            gt_l.append(batch["ref_point_motion"].numpy())
            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)
    pa, pb = np.concatenate(pa_l), np.concatenate(pb_l)
    la, lb = np.concatenate(la_l), np.concatenate(lb_l)
    gt = np.concatenate(gt_l)
    thresholds = sorted(set(args.thresholds))

    def best_at(prob):
        return max(((iou_at(prob, gt, th), th) for th in thresholds),
                   key=lambda x: x[0][0])

    rows = []
    for fusion in ("prob", "logit"):
        for w in args.weights:
            if not 0 <= w <= 1:
                raise ValueError("ensemble weights must be in [0, 1]")
            if fusion == "prob":
                pred = w * pa + (1.0 - w) * pb
            else:
                margin = w * la + (1.0 - w) * lb
                pred = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
            (iou, prec, rec), th = best_at(pred)
            rows.append((iou, fusion, w, th, prec, rec))

    print(f"\n=== ensemble on val seq {d['val_sequences']} ===")
    print(f"{'fusion':>8} {'w(A)':>6} {'best IoU':>9} {'@th':>5} "
          f"{'Prec':>8} {'Rec':>8}")
    for iou, fusion, w, th, prec, rec in rows:
        print(f"{fusion:>8} {w:6.2f} {iou:9.4f} {th:5.2f} "
              f"{prec:8.4f} {rec:8.4f}")
    iou, fusion, w, th, prec, rec = max(rows)
    print(f"\nBEST: IoU={iou:.4f} fusion={fusion} w(A)={w:.2f} "
          f"threshold={th:.2f} P={prec:.4f} R={rec:.4f}")


if __name__ == "__main__":
    main()
