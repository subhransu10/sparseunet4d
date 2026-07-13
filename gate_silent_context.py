"""PHASE-2 KILL GATE (CPU only, no training, no checkpoint).

Hypothesis under test: range-image SPATIAL CONTEXT carries signal on "silent"
movers (per-point |residual| < 0.2 m) that per-point residuals lack.

Method (val-08, GT poses, GT motion labels):
  - Build residual images Delta=1,2,3 (reuses spherical_project).
  - Pixel classes: SILENT-MOVING (GT moving, per-point |res|<0.2 for ALL Delta),
    LOUD-MOVING (rest of moving), STATIC.
  - Patch-context features per pixel (5x9 window): mean|res|, max|res|,
    std over Delta, valid-fraction.
  - Report AUROC (rank-based, numpy) of each feature + a simple sum-combo for
    SILENT-MOVING vs STATIC.

PRE-REGISTERED DECISION:
  best AUROC <= 0.60  -> KILL Phase 2 (CNN has nothing to amplify)
  best AUROC >= 0.70  -> PROCEED to T-ladder
  in between          -> proceed only with reduced expectations / cheap 5k probe

Usage:
  PYTHONPATH=~/sparseunet4d python3 gate_silent_context.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml [--stride 5] [--max-frames 400]
"""
import os, sys, argparse, yaml
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets.semantickitti import _read_scan, _read_label, _transform
from sparseunet4d.datasets.label_map import split_label, to_motion_labels
from sparseunet4d.datasets.poses import build_pose_provider
from sparseunet4d.datasets.residual_features import spherical_project

H, W = 64, 2048
SILENT_TH = 0.2
CLIP = 3.0
WIN = (5, 9)          # (rows, cols) context window


def auroc(pos, neg):
    """Rank-based AUROC, numpy only. pos/neg: 1D feature arrays."""
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x, kind="mergesort")) + 1.0  # ranks 1..n (ties approx)
    # proper tie handling via average ranks:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def frame_features(root, seq, ref, provider, n_frames=4):
    seq_dir = os.path.join(root, f"{seq:02d}")
    bin_p = os.path.join(seq_dir, "velodyne", f"{ref:06d}.bin")
    lab_p = os.path.join(seq_dir, "labels", f"{ref:06d}.label")
    if not os.path.exists(lab_p) or ref < n_frames - 1:
        return None
    scan = _read_scan(bin_p)
    xyz = scan[:, :3]
    sem_raw, _ = split_label(_read_label(lab_p))
    mot = to_motion_labels(sem_raw)

    now_img, u, v, r_now, valid = spherical_project(xyz, H, W)
    K = n_frames - 1
    res_imgs = np.zeros((K, H, W), np.float32)
    res_pts = np.zeros((len(xyz), K), np.float32)
    vmask = np.zeros((K, H, W), bool)
    for k in range(1, n_frames):
        past = _read_scan(os.path.join(seq_dir, "velodyne", f"{ref-k:06d}.bin"))[:, :3]
        past = _transform(past, provider.relative(ref - k, ref)).astype(np.float32)
        p_img, *_ = spherical_project(past, H, W)
        ok = np.isfinite(now_img) & np.isfinite(p_img)
        res_imgs[k-1][ok] = np.clip(now_img[ok] - p_img[ok], -CLIP, CLIP)
        vmask[k-1] = ok
        rp = p_img[v, u]
        hp = np.isfinite(rp) & valid
        res_pts[hp, k-1] = np.clip(r_now[hp] - rp[hp], -CLIP, CLIP)

    # patch-context features per pixel
    a = np.abs(res_imgs)                       # (K,H,W)
    mean_abs = ndimage.uniform_filter(a.mean(0), WIN)
    max_abs = ndimage.maximum_filter(a.max(0), WIN)
    std_t = ndimage.uniform_filter(res_imgs.std(0), WIN)
    valid_fr = ndimage.uniform_filter(vmask.mean(0).astype(np.float32), WIN)
    feats_img = np.stack([mean_abs, max_abs, std_t, valid_fr], -1)  # (H,W,4)

    # per-point pixel labels
    silent_pt = (np.abs(res_pts) < SILENT_TH).all(1)
    F = feats_img[v[valid], u[valid]]
    m = mot[valid].astype(bool)
    s = silent_pt[valid]
    return F, m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dataset"]; root = d["root"]; seq = d["val_sequences"][0]
    nF = d.get("n_frames", 4)
    provider = build_pose_provider(os.path.join(root, f"{seq:02d}"), "gt", 0, 0, 0)

    n = len(os.listdir(os.path.join(root, f"{seq:02d}", "velodyne")))
    frames = list(range(nF, n, args.stride))[:args.max_frames]

    Fs, sil, loud, sta = [], [], [], []
    rng = np.random.default_rng(0)
    for i, ref in enumerate(frames):
        out = frame_features(root, seq, ref, provider, nF)
        if out is None:
            continue
        F, m, s = out
        sil.append(F[m & s])
        loud.append(F[m & ~s])
        stat = F[~m]
        keep = rng.choice(len(stat), min(len(stat), 4000), replace=False)
        sta.append(stat[keep])
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    sil, loud, sta = map(np.concatenate, (sil, loud, sta))
    print(f"\npixels: silent-moving={len(sil)}  loud-moving={len(loud)}  "
          f"static(sampled)={len(sta)}")
    names = ["mean|res| (5x9)", "max|res| (5x9)", "std_t (5x9)", "valid_frac"]
    print(f"\n=== AUROC: SILENT-moving vs static ===")
    best = 0.0
    for j, nm in enumerate(names):
        a = auroc(sil[:, j], sta[:, j]); best = max(best, a)
        print(f"  {nm:>18}: {a:.4f}")
    combo = lambda X: X[:, 0] + X[:, 1] + X[:, 2]
    a = auroc(combo(sil), combo(sta)); best = max(best, a)
    print(f"  {'sum-combo':>18}: {a:.4f}")
    print(f"\n(sanity) LOUD-moving vs static, max|res|: "
          f"{auroc(loud[:, 1], sta[:, 1]):.4f}  <- should be high (~0.9+)")
    verdict = ("PROCEED" if best >= 0.70 else
               "KILL" if best <= 0.60 else "GREY ZONE - cheap 5k probe only")
    print(f"\nbest AUROC = {best:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()