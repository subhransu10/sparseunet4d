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


def threshold_counts(scores, positive, thresholds):
    """Return exact positive/negative prediction counts without N*T arrays."""
    # bin k means exactly k sorted thresholds are <= the score.  Therefore the
    # number predicted positive at threshold j is sum(hist[j + 1:]).
    bins = np.searchsorted(thresholds, scores, side="right")
    pos_hist = np.bincount(bins[positive], minlength=len(thresholds) + 1)
    neg_hist = np.bincount(bins[~positive], minlength=len(thresholds) + 1)
    pos_ge = np.cumsum(pos_hist[::-1], dtype=np.int64)[::-1][1:]
    neg_ge = np.cumsum(neg_hist[::-1], dtype=np.int64)[::-1][1:]
    return pos_ge, neg_ge


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

    thresholds = np.asarray(sorted(set(args.thresholds)), dtype=np.float32)
    weights = np.asarray(args.weights, dtype=np.float32)
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("ensemble weights must be in [0, 1]")

    # [fusion, weight, threshold]. Accumulating confusion counts per frame
    # avoids retaining hundreds of millions of point predictions in RAM.
    tp = np.zeros((2, len(weights), len(thresholds)), dtype=np.int64)
    fp = np.zeros_like(tp)
    total_pos = 0
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
            pa, pb = prob_a[rpv], prob_b[rpv]
            la, lb = margin_a[rpv], margin_b[rpv]
            gt = batch["ref_point_motion"].numpy()
            valid = gt != -1
            positive = gt[valid] == 1
            total_pos += int(positive.sum())
            pa, pb, la, lb = pa[valid], pb[valid], la[valid], lb[valid]

            for wi, w in enumerate(weights):
                scores = w * pa + (1.0 - w) * pb
                batch_tp, batch_fp = threshold_counts(
                    scores, positive, thresholds)
                tp[0, wi] += batch_tp
                fp[0, wi] += batch_fp

                margin = w * la + (1.0 - w) * lb
                scores = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
                batch_tp, batch_fp = threshold_counts(
                    scores, positive, thresholds)
                tp[1, wi] += batch_tp
                fp[1, wi] += batch_fp
            if bi % 500 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    rows = []
    for fi, fusion in enumerate(("prob", "logit")):
        for wi, w in enumerate(weights):
            fn = total_pos - tp[fi, wi]
            denom = tp[fi, wi] + fp[fi, wi] + fn
            ious = tp[fi, wi] / np.maximum(denom, 1)
            ti = int(np.argmax(ious))
            tpi, fpi, fni = (int(tp[fi, wi, ti]), int(fp[fi, wi, ti]),
                             int(fn[ti]))
            iou = float(ious[ti])
            prec = tpi / max(tpi + fpi, 1)
            rec = tpi / max(tpi + fni, 1)
            rows.append((iou, fusion, float(w), float(thresholds[ti]),
                         prec, rec))

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
