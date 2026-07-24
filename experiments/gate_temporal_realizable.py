"""PHASE-3 REALIZABILITY GATE -- and, if it passes, the method itself.

The temporal oracle (GT-id association) says ceiling 0.7087 @ K=10. This script
replaces GT association with what a real system has: PREDICTED confident
clusters, linked into tracklets in the world frame, constant-velocity
extrapolated to the silent frame. Because it claims clusters exactly like
propagation v2, a passing run here IS the final method (post-hoc, no training)
-- the "gate" and "implementation" collapse into one experiment.

Pass 1 (GPU, one val-08 sweep): per frame cache, in WORLD coordinates
(frame-0 anchor, GT poses):
  conf_clusters  : centroid + nvox of clusters of confident predicted-moving
                   voxels (prob>=0.5), 0.4 m linkage
  silent_clusters: movable-sem clusters with max prob < 0.5: centroid, total
                   nvox, GT-moving nvox (so FP cost of a wrong claim is exact)
Plus frame TP/FP/FN for the baseline consistency check (must print 0.6235).

Pass 2 (CPU, instant, re-runnable via --analyze-only):
  1. Greedy tracklet linking of conf_clusters across frames
     (NN <= LINK_M per frame step, coasting up to COAST frames).
  2. For each silent cluster at frame t: claim it if any tracklet,
     CV-extrapolated to t (obs within +-K), lands within GATE_M.
  3. Tally EXACT IoU/precision deltas: claimed GT-moving voxels -> recovered
     FN; claimed GT-static voxels -> added FP. Sweep K and GATE_M.

PRE-REGISTERED (from the FP-budget note):
  realized IoU < 0.655 or precision < 0.85 at every (K, GATE_M) -> KILL.
  realized IoU > 0.6430 with precision >= 0.85 -> this is propagation v3;
  best cell becomes the reported configuration. Oracle ceiling: 0.7087.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/gate_temporal_realizable.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml \
    --ckpt ~/sparseunet4d/runs/residual_v2/best.pt \
    --cache ~/sparseunet4d/runs/residual_v2/temporal_realizable.npz
  python3 gate_temporal_realizable.py --analyze-only --cache .../temporal_realizable.npz
"""
import os, sys, argparse, yaml
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))

TH = 0.5
LINK_VOX = 4                       # 0.4 m cluster linkage (matches v2)
MIN_CLUSTER = 5
MOVABLE_SEM = {1, 2, 3, 4, 5, 6, 7, 8}
LINK_M = 2.0                       # tracklet frame-to-frame association gate
COAST = 5                          # tracklet survives this many missed frames
K_SWEEP = [3, 5, 10]
GATE_SWEEP = [1.0, 1.5, 2.0]       # claim radius (m) after CV extrapolation


def components(vox):
    if len(vox) == 0:
        return np.zeros(0, np.int64)
    c = (vox // LINK_VOX).astype(np.int64); c -= c.min(0)
    grid = np.zeros(c.max(0) + 1, dtype=bool)
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    return lab[c[:, 0], c[:, 1], c[:, 2]]


# ---------------------------------------------------------------------------
def build_cache(args):
    import torch
    from torch.utils.data import DataLoader
    from sparseunet4d.datasets import SemanticKITTI4D, me_collate
    from sparseunet4d.datasets.semantickitti import _transform
    from sparseunet4d.datasets.poses import build_pose_provider
    from sparseunet4d.models.backend import backend
    from sparseunet4d.models.model import SparseUNet4D

    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vox_sz = d["voxel_size"]; seq = d["val_sequences"][0]
    provider = build_pose_provider(
        os.path.join(d["root"], f"{seq:02d}"), "gt", 0, 0, 0)

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        vox_sz, d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20),
        base=m.get("base", 32), n_stages=m.get("n_stages", 2),
        use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(
        ck["model"] if "model" in ck else ck, strict=False)
    assert all(k.startswith("offset_head") for k in missing), missing
    assert not unexpected, unexpected

    conf_rows, sil_rows = [], []       # (frame, cx, cy, cz, nvox[, nmov])
    tp = fp = fn = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            _, ref = batch["meta"][0]
            coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
            if backend() == "me":
                import MinkowskiEngine as ME
                x = ME.SparseTensor(feats, coordinates=coords)
            else:
                from sparseunet4d.models.backend import ST
                x = ST(feats, coords)
            out = model(x)
            prob = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
            sem_pred = out["semantic_logits"].argmax(1).cpu().numpy()
            gt = batch["motion"].numpy()
            cnp = batch["coords"].numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup]; sp = sem_pred[sup]
            vx = cnp[sup][:, 1:4].astype(np.float64) * vox_sz  # sensor metres
            conf = pm >= TH
            tp += int((conf & (g == 1)).sum())
            fp += int((conf & (g == 0)).sum())
            fn += int((~conf & (g == 1)).sum())

            T = provider.relative(ref, 0)                      # sensor->world

            # confident clusters (detections)
            cv = (vx[conf] / vox_sz).astype(np.int64)
            lab = components(cv)
            for lid in np.unique(lab):
                s = lab == lid
                if s.sum() < MIN_CLUSTER:
                    continue
                cw = _transform(vx[conf][s].mean(0)[None], T)[0]
                conf_rows.append((ref, cw[0], cw[1], cw[2], int(s.sum())))

            # silent movable-sem clusters (claim candidates)
            movable = np.isin(sp, list(MOVABLE_SEM))
            mv = (vx[movable] / vox_sz).astype(np.int64)
            lab = components(mv)
            pmov = pm[movable]; gmov = g[movable]
            for lid in np.unique(lab):
                s = lab == lid
                if s.sum() < MIN_CLUSTER or pmov[s].max() >= TH:
                    continue                                    # not silent
                cw = _transform(vx[movable][s].mean(0)[None], T)[0]
                sil_rows.append((ref, cw[0], cw[1], cw[2],
                                 int(s.sum()), int((gmov[s] == 1).sum())))
            if bi % 400 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    np.savez_compressed(args.cache,
        conf=np.array(conf_rows, np.float64),
        sil=np.array(sil_rows, np.float64), tp=tp, fp=fp, fn=fn)
    print(f"cached {len(conf_rows)} confident clusters, "
          f"{len(sil_rows)} silent clusters -> {args.cache}")
    print(f"baseline voxel IoU sanity: {tp/max(tp+fp+fn,1):.4f} (expect 0.6235)")


# ---------------------------------------------------------------------------
def build_tracklets(conf):
    """conf rows: (frame, x, y, z, nvox). Greedy NN linking with coasting.
    Returns list of dicts {frames: [...], pos: [...] } sorted by frame."""
    by_f = {}
    for r in conf:
        by_f.setdefault(int(r[0]), []).append(r[1:4])
    tracks = []          # each: {"f": [frames], "p": [np.array(3)]}
    active = []
    for f in sorted(by_f):
        dets = [np.asarray(q) for q in by_f[f]]
        used = [False] * len(dets)
        for tr in active:
            dt = f - tr["f"][-1]
            if dt > COAST:
                continue
            if len(tr["p"]) >= 2:
                v = (tr["p"][-1] - tr["p"][-2]) / max(tr["f"][-1] - tr["f"][-2], 1)
            else:
                v = np.zeros(3)
            pred = tr["p"][-1] + v * dt
            best, bd = -1, LINK_M * dt if dt > 1 else LINK_M
            for i, q in enumerate(dets):
                if used[i]:
                    continue
                dist = np.linalg.norm(q - pred)
                if dist < bd:
                    best, bd = i, dist
            if best >= 0:
                used[best] = True
                tr["f"].append(f); tr["p"].append(dets[best])
        for i, q in enumerate(dets):
            if not used[i]:
                tracks.append({"f": [f], "p": [q]})
        active = [t for t in tracks if f - t["f"][-1] <= COAST]
    return tracks


def analyze(args):
    z = np.load(args.cache)
    conf, sil = z["conf"], z["sil"]
    tp, fp, fn = int(z["tp"]), int(z["fp"]), int(z["fn"])
    base_iou = tp / max(tp + fp + fn, 1)
    print(f"baseline: IoU={base_iou:.4f} TP={tp} FP={fp} FN={fn}")
    print(f"confident clusters={len(conf)}  silent clusters={len(sil)} "
          f"(GT-moving voxels in silent: {int(sil[:,5].sum())}, "
          f"static: {int((sil[:,4]-sil[:,5]).sum())})")

    tracks = build_tracklets(conf)
    print(f"tracklets: {len(tracks)} "
          f"(len>=3: {sum(len(t['f'])>=3 for t in tracks)})")
    # index tracklet observations by frame for fast lookup
    obs = {}
    for ti, tr in enumerate(tracks):
        for j, f in enumerate(tr["f"]):
            obs.setdefault(f, []).append((ti, j))

    def extrapolate(tr, j, t):
        """CV from observation j to frame t (velocity from nearest neighbor obs)."""
        f0, p0 = tr["f"][j], tr["p"][j]
        if len(tr["f"]) >= 2:
            j2 = j - 1 if j > 0 else j + 1
            v = (tr["p"][j] - tr["p"][j2]) / (tr["f"][j] - tr["f"][j2])
        else:
            v = np.zeros(3)
        return p0 + v * (t - f0)

    hdr = f"{'K':>4} {'gate[m]':>8} {'claimed':>8} {'recov':>8} {'+FP':>8} " \
          f"{'IoU':>8} {'P':>8}"
    print("\n" + hdr)
    best = (base_iou, None)
    for K in K_SWEEP:
        for G in GATE_SWEEP:
            rec = add_fp = n_claim = 0
            for r in sil:
                t, c = int(r[0]), r[1:4]
                hit = False
                for k in range(0, K + 1):
                    for f2 in ({t - k, t + k} if k else {t}):
                        for ti, j in obs.get(f2, []):
                            pred = extrapolate(tracks[ti], j, t)
                            if np.linalg.norm(pred - c) < G:
                                hit = True; break
                        if hit: break
                    if hit: break
                if hit:
                    n_claim += 1
                    rec += int(r[5]); add_fp += int(r[4] - r[5])
            iou = (tp + rec) / max(tp + fp + fn + add_fp, 1)
            prec = (tp + rec) / max(tp + rec + fp + add_fp, 1)
            print(f"{K:>4} {G:>8.1f} {n_claim:>8d} {rec:>8d} {add_fp:>8d} "
                  f"{iou:>8.4f} {prec:>8.4f}")
            if iou > best[0] and prec >= 0.85:
                best = (iou, (K, G, prec))

    print(f"\nbars: beat 0.6430; kill if best IoU < 0.655 or P < 0.85. "
          f"oracle ceiling 0.7087.")
    if best[1] is None:
        print(f"best realized IoU <= baseline under P>=0.85  ->  KILL")
    else:
        K, G, prec = best[1]
        verdict = ("KILL (below 0.655 floor)" if best[0] < 0.655 else
                   "PROPAGATION v3: report this configuration")
        print(f"best: IoU={best[0]:.4f} P={prec:.4f} @ K={K}, gate={G}m "
              f"->  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        assert args.config and args.ckpt, "--config/--ckpt required for pass 1"
        build_cache(args)
    analyze(args)