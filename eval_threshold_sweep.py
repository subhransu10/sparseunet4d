"""Threshold sweep for the moving-class decision on SemanticKITTI-MOS val.

Same protocol as eval_mos_official.py, but instead of argmax (cutoff 0.5) it
sweeps softmax cutoffs and reports P/R/IoU at each. One forward pass per batch;
all thresholds accumulated simultaneously. Answers: does the model KNOW about
more movers than it declares at 0.5?

Usage (same env pattern as eval_mos_official.py):
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_threshold_sweep.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml \
    --ckpt ~/sparseunet4d/runs/residual_v2/best.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    residual_feats = d.get("residual_feats", True)
    in_ch = 1 + (n_frames - 1) if residual_feats else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    print(f"loaded {args.ckpt}  (best_moving_iou={ck.get('best_moving_iou')})")

    K = len(THRESHOLDS)
    tp = np.zeros(K, np.int64); fp = np.zeros(K, np.int64); fn = np.zeros(K, np.int64)

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
            gt = batch["motion"].numpy()
            mask = gt != -1
            pm, g = prob[mask], gt[mask]
            gpos = g == 1
            for k, th in enumerate(THRESHOLDS):
                pr = pm >= th
                tp[k] += int((pr & gpos).sum())
                fp[k] += int((pr & ~gpos).sum())
                fn[k] += int((~pr & gpos).sum())
            if bi % 200 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)

    print(f"\n=== threshold sweep, val seq {d['val_sequences']} ===")
    print(f"{'thresh':>7} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'TP':>9} {'FP':>9} {'FN':>9}")
    best_k = 0; best_iou = -1
    for k, th in enumerate(THRESHOLDS):
        iou = tp[k] / max(tp[k] + fp[k] + fn[k], 1)
        prec = tp[k] / max(tp[k] + fp[k], 1)
        rec = tp[k] / max(tp[k] + fn[k], 1)
        star = ""
        if iou > best_iou: best_iou, best_k, star = iou, k, ""
        print(f"{th:7.2f} {iou:8.4f} {prec:8.4f} {rec:8.4f} "
              f"{tp[k]:9d} {fp[k]:9d} {fn[k]:9d}")
    print(f"\nbest IoU {best_iou:.4f} @ threshold {THRESHOLDS[best_k]}")
    print("Interpretation: if best threshold < 0.5 with IoU clearly above the "
          "0.5 row, the model knows more movers than argmax declares "
          "(operating-point headroom). If 0.5 is already best, remaining FNs "
          "are low-confidence everywhere -> information limit, not calibration.")


if __name__ == "__main__":
    main()