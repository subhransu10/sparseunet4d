"""PHASE-2 KILL GATE v2.1 -- image-plane displacement, REPAIRED.

v2.0 was invalid: its positive control failed (loud-mover AUROC ~0.5 vs 0.78
for v1's max|res|). Root cause: FILL=80 painted into empty pixels dominated
window SADs; occupancy flicker looks the same for static and moving regions.

Fixes:
  1. MASKED SAD: |now - roll(past, s)| only where BOTH pixels are valid;
     window SAD = sum(diff*mask)/sum(mask). Windows with < MIN_VALID mutual
     validity are neutralized (all shifts equal -> shift=0, gain=0).
  2. churn = XOR / (occupied-union) per window, not raw XOR.
  3. HARD SANITY CHECK: loud-mover AUROC must clear 0.70 on shift_mag or
     sad_gain, else the script prints GATE INVALID and issues NO verdict.
     A broken gate can no longer produce a KILL/PROCEED.

Decision thresholds unchanged (pre-registered): silent-vs-static best AUROC
<= 0.60 KILL (Phase 2 closed), >= 0.70 PROCEED (raw-temporal-stack branch).

Usage:
  PYTHONPATH=~/sparseunet4d python3 gate_v2_displacement.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml [--stride 5] [--max-frames 400]
  Self-test (sparse synthetic, no data): python3 gate_v2_displacement.py --self-test
"""
import os, sys, argparse, yaml
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))

H, W = 64, 2048
SILENT_TH = 0.2
CLIP = 3.0
WIN = (5, 9)
MAX_SHIFT = 8
MIN_VALID = 0.30      # min mutual-validity fraction for a window to vote


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = len(pos)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg))


def displacement_features(now_img, past_imgs):
    """Masked features. Returns (H,W,3): [churn, shift_mag, sad_gain]."""
    occ_now = np.isfinite(now_img)
    a = np.where(occ_now, now_img, 0.0)
    K = len(past_imgs)
    fx = np.zeros((H, W)); fs = np.zeros((H, W)); fg = np.zeros((H, W))
    uf = lambda z: ndimage.uniform_filter(z.astype(np.float32), WIN, mode="wrap")
    for p in past_imgs:
        occ_p = np.isfinite(p)
        b = np.where(occ_p, p, 0.0)
        # normalized churn
        union = uf(occ_now | occ_p)
        fx += np.where(union > 0, uf(occ_now ^ occ_p) / np.maximum(union, 1e-6), 0)
        # masked SAD over horizontal shifts
        S = 2 * MAX_SHIFT + 1
        sads = np.empty((S, H, W), np.float32)
        vfrac = np.empty((S, H, W), np.float32)
        for si, s in enumerate(range(-MAX_SHIFT, MAX_SHIFT + 1)):
            bs = np.roll(b, s, axis=1); ms = np.roll(occ_p, s, axis=1)
            m = (occ_now & ms).astype(np.float32)
            d = np.minimum(np.abs(a - bs), CLIP) * m
            den = uf(m)
            sads[si] = np.where(den > 1e-6, uf(d) / np.maximum(den, 1e-6), CLIP)
            vfrac[si] = den
        usable = vfrac.min(0) >= MIN_VALID     # every shift must have support
        best = sads.argmin(0)
        smag = np.abs(best - MAX_SHIFT).astype(np.float32)
        gain = sads[MAX_SHIFT] - np.take_along_axis(sads, best[None], 0)[0]
        fs += np.where(usable, smag, 0.0)
        fg += np.where(usable, gain, 0.0)
    return np.stack([fx / K, fs / K, fg / K], -1)


def run_dataset(args):
    from sparseunet4d.datasets.semantickitti import _read_scan, _read_label, _transform
    from sparseunet4d.datasets.label_map import split_label, to_motion_labels
    from sparseunet4d.datasets.poses import build_pose_provider
    from sparseunet4d.datasets.residual_features import spherical_project

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dataset"]; root = d["root"]; seq = d["val_sequences"][0]
    nF = d.get("n_frames", 4)
    provider = build_pose_provider(os.path.join(root, f"{seq:02d}"), "gt", 0, 0, 0)
    seq_dir = os.path.join(root, f"{seq:02d}")
    n = len(os.listdir(os.path.join(seq_dir, "velodyne")))
    frames = list(range(nF, n, args.stride))[:args.max_frames]

    rng = np.random.default_rng(0)
    sil, loud, sta = [], [], []
    for i, ref in enumerate(frames):
        lab_p = os.path.join(seq_dir, "labels", f"{ref:06d}.label")
        if not os.path.exists(lab_p):
            continue
        scan = _read_scan(os.path.join(seq_dir, "velodyne", f"{ref:06d}.bin"))
        xyz = scan[:, :3]
        sem_raw, _ = split_label(_read_label(lab_p))
        mot = to_motion_labels(sem_raw).astype(bool)
        now_img, u, v, r_now, valid = spherical_project(xyz, H, W)

        past_imgs, res_pts = [], np.zeros((len(xyz), nF - 1), np.float32)
        for k in range(1, nF):
            past = _read_scan(os.path.join(seq_dir, "velodyne",
                                           f"{ref-k:06d}.bin"))[:, :3]
            past = _transform(past, provider.relative(ref - k, ref)).astype(np.float32)
            p_img, *_ = spherical_project(past, H, W)
            past_imgs.append(p_img)
            rp = p_img[v, u]
            hp = np.isfinite(rp) & valid
            res_pts[hp, k-1] = np.clip(r_now[hp] - rp[hp], -CLIP, CLIP)

        F = displacement_features(now_img, past_imgs)[v[valid], u[valid]]
        silent = (np.abs(res_pts) < SILENT_TH).all(1)[valid]
        m = mot[valid]
        sil.append(F[m & silent]); loud.append(F[m & ~silent])
        stat = F[~m]
        keep = rng.choice(len(stat), min(len(stat), 4000), replace=False)
        sta.append(stat[keep])
        if i % 25 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    sil, loud, sta = map(np.concatenate, (sil, loud, sta))
    print(f"\npixels: silent-moving={len(sil)}  loud-moving={len(loud)}  "
          f"static(sampled)={len(sta)}")
    names = ["churn", "shift_mag", "sad_gain"]

    # ---- HARD SANITY (positive control) first ------------------------------
    loud_a = [auroc(loud[:, j], sta[:, j]) for j in range(3)]
    print("\n(positive control) LOUD-moving vs static:")
    for nm, a in zip(names, loud_a):
        print(f"  {nm:>10}: {a:.4f}")
    if max(loud_a[1], loud_a[2]) < 0.70:
        print("\nGATE INVALID: positive control failed "
              f"(best motion feature {max(loud_a[1], loud_a[2]):.3f} < 0.70). "
              "NO verdict issued -- the features cannot see known movers; "
              "fix the gate before interpreting silent-mover numbers.")
        return

    print("\n=== AUROC: SILENT-moving vs static ===")
    best = 0.0
    for j, nm in enumerate(names):
        a = auroc(sil[:, j], sta[:, j]); best = max(best, a)
        print(f"  {nm:>10}: {a:.4f}")
    combo = lambda X: X[:, 0] + X[:, 1] / MAX_SHIFT + X[:, 2] / CLIP
    a = auroc(combo(sil), combo(sta)); best = max(best, a)
    print(f"  {'combo':>10}: {a:.4f}")
    verdict = ("PROCEED (raw-temporal-stack branch)" if best >= 0.70 else
               "KILL -- Phase 2 closed (legitimate: positive control passed)"
               if best <= 0.60 else "GREY ZONE -- 5k probe max")
    print(f"\nbest AUROC = {best:.4f}  ->  {verdict}")


def self_test():
    """SPARSE synthetic (60% dropout, jittered wall): tangential mover at
    constant range must fire shift/gain; static+dropout must stay quiet."""
    rng = np.random.default_rng(1)
    def scene(x0):
        img = np.full((H, W), np.inf)
        occ = rng.random((H, W)) > 0.6          # 40% occupied, like real
        img[occ] = 20.0 + rng.normal(0, 0.05, occ.sum())
        ys, xs = np.mgrid[30:36, x0:x0+40]
        keep = rng.random(ys.size) > 0.3        # object also sparse
        img[ys.ravel()[keep], xs.ravel()[keep]] = 8.0
        return img
    now = scene(1000)
    past = [scene(1000 - 4*k) for k in (1, 2, 3)]
    F = displacement_features(now, past)
    obj = F[31:35, 1005:1035].reshape(-1, 3)
    bg = F[5:20, 100:1900].reshape(-1, 3)
    print("object  churn/shift/gain:", obj.mean(0).round(3))
    print("static  churn/shift/gain:", bg.mean(0).round(3))
    a_shift = auroc(obj[:, 1], bg[:, 1]); a_gain = auroc(obj[:, 2], bg[:, 2])
    print(f"AUROC obj-vs-static: shift={a_shift:.3f} gain={a_gain:.3f}")
    assert max(a_shift, a_gain) > 0.85, "self-test failed under sparsity"
    assert bg[:, 1].mean() < 1.0, "static background shift too noisy"
    print("SELF-TEST PASS (sparse): tangential mover detected, static quiet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.config, "--config required (or use --self-test)"
        run_dataset(args)