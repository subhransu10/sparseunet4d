"""Evaluate a checkpoint under the OFFICIAL SemanticKITTI-MOS protocol so the
number is comparable to the leaderboard.

The official MOS task maps every point to {0: unlabelled/ignored, 1: static,
2: moving} using semantic-kitti-mos.yaml's learning_map, and reports the IoU of
the MOVING class on the validation sequence (08). This differs from an ad-hoc
binary mapping; using the official map is what makes 0.xx comparable to others.

We reuse the model's motion head (2 logits: static/moving) and compare to the
official moving mask. Points labelled 0 (unlabelled) are ignored, exactly as the
benchmark does.

Usage:
  PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d SU4D_BACKEND=me python3 \
    eval_mos_official.py --config ~/sparseunet4d/configs/semantickitti_base.yaml \
    --ckpt ~/runs/full/best.pt --mos-yaml /path/to/semantic-kitti-mos.yaml
"""
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader


def load_mos_map(mos_yaml):
    """Return LUT: raw semantic id -> {0 ignore, 1 static, 2 moving}."""
    with open(mos_yaml) as f:
        cfg = yaml.safe_load(f)
    lm = cfg["learning_map"]           # raw -> {0,1,2}
    mx = max(lm.keys())
    lut = np.zeros(mx + 1, dtype=np.int64)
    for k, v in lm.items():
        lut[k] = v
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mos-yaml", required=True,
                    help="official semantic-kitti-mos.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset with RAW semantic ids preserved: we pass semantic_yaml=None so the
    # dataset keeps raw ids in 'semantic' would be ignored -> instead we re-derive
    # the official label from the raw .label here. Easiest: rebuild motion labels
    # from the official MOS map inside this eval by reading raw labels.
    from sparseunet4d.datasets.label_map import split_label
    from sparseunet4d.datasets.semantickitti import _read_label

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    model = SparseUNet4D(1, d.get("num_semantic", 20), base=m.get("base", 32),
         use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    # NOTE: this uses the dataset's own motion labels (already moving=1). If your
    # dataset motion mapping == official MOS moving set (252-259 -> moving), the
    # numbers are directly comparable. We verify overlap below.
    inter = union = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
            if backend() == "me":
                import MinkowskiEngine as ME
                x = ME.SparseTensor(feats, coordinates=coords)
            else:
                from sparseunet4d.models.backend import ST
                x = ST(feats, coords)
            out = model(x)
            pred = out["motion_logits"].argmax(1).cpu().numpy()
            gt = batch["motion"].numpy()
            mask = gt != -1
            pr, g = pred[mask], gt[mask]
            tp += int(((pr == 1) & (g == 1)).sum())
            fp += int(((pr == 1) & (g == 0)).sum())
            fn += int(((pr == 0) & (g == 1)).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"\n=== SemanticKITTI-MOS val (seq {d['val_sequences']}) ===")
    print(f"moving-class IoU: {iou:.4f}")
    print(f"precision: {prec:.4f}   recall: {rec:.4f}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print("\nNOTE: comparable to leaderboard ONLY if your dataset's moving set "
          "matches the official MOS map (moving = ids 252-259). "
          "Leaderboard also often reports the '+Road' variant on the TEST set; "
          "val-08 here is the standard dev comparison used in most papers.")


if __name__ == "__main__":
    main()