"""Score a baseline's predicted .label files against SemanticKITTI GT (MOS).

Made for 4DMOS-style outputs (predictions saved per scan as .label with raw
ids 251=moving / 9=static), but --moving-pred-ids makes it generic. Evaluates
point-level moving IoU; --range-clip restricts to |x|,|y|,|z| < R to match our
protocol (pass 51.2 when comparing curves against our numbers; omit for the
baseline's native full-cloud protocol — for the drift FIGURE we plot each
method normalized to its own clean value, so either is defensible, just be
consistent across levels).

Usage:
  python3 eval_pred_labels.py \
      --gt-seq "<root>/sequences/08" \
      --pred-dir 4dmos_out/drift_r0.5_t0.1/sequences/08/predictions \
      [--range-clip 51.2]
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparseunet4d.datasets.label_map import MOVING_IDS, split_label
from sparseunet4d.datasets.semantickitti import _read_scan, _read_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-seq", required=True, help=".../sequences/08")
    ap.add_argument("--pred-dir", required=True,
                    help="dir of predicted .label files (000000.label ...)")
    ap.add_argument("--moving-pred-ids", type=int, nargs="+", default=[251],
                    help="raw ids meaning 'moving' in the predictions")
    ap.add_argument("--range-clip", type=float, default=None,
                    help="e.g. 51.2 to match our in-range protocol")
    args = ap.parse_args()

    lab_dir = os.path.join(args.gt_seq, "labels")
    velo_dir = os.path.join(args.gt_seq, "velodyne")
    preds = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".label"))
    assert preds, f"no .label files in {args.pred_dir}"

    tp = fp = fn = 0
    for k, f in enumerate(preds):
        gt_p = os.path.join(lab_dir, f)
        if not os.path.exists(gt_p):
            continue
        sem_raw, _ = split_label(_read_label(gt_p))
        pr_raw, _ = split_label(_read_label(os.path.join(args.pred_dir, f)))
        assert len(sem_raw) == len(pr_raw), f"{f}: length mismatch"
        keep = np.ones(len(sem_raw), bool)
        if args.range_clip is not None:
            xyz = _read_scan(os.path.join(velo_dir, f.replace(".label", ".bin")))[:, :3]
            keep = np.all(np.abs(xyz) < args.range_clip, axis=1)
        g = np.isin(sem_raw, list(MOVING_IDS)) & keep
        p = np.isin(pr_raw, args.moving_pred_ids) & keep
        tp += int((p & g).sum()); fp += int((p & ~g).sum())
        fn += int((~p & g).sum())
        if k % 500 == 0:
            print(f"  {k}/{len(preds)}", flush=True)

    iou = tp / max(tp + fp + fn, 1)
    print(f"\n=== {args.pred_dir} ===")
    print(f"scans scored: {len(preds)}  range_clip={args.range_clip}")
    print(f"moving IoU: {iou:.4f}   P={tp/max(tp+fp,1):.4f} "
          f"R={tp/max(tp+fn,1):.4f}   TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
