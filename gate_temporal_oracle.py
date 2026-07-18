"""PHASE-3 ORACLE GATE: are silent-FN instances loud somewhere nearby in time?

Pass 1 (GPU, one val-08 sweep, same cost as your other evals): for every frame,
run the residual_v2 checkpoint and record a tiny per-instance table:
  (frame, instance_key, n_voxels, n_confident, n_pred_pos)
where instance_key = sem_raw * 100000 + inst (SemanticKITTI instance ids are
temporally consistent within a sequence, so no spatial registration needed),
restricted to GT-MOVING instances. Also frame-level TP/FP/FN for the baseline.
Cached to an .npz; pass 2 is instant and re-runnable with --analyze-only.

Pass 2 (CPU): an instance is SILENT in frame t if n_confident == 0 (matches
your FN characterization). For each silent (t, inst), scan t±1..±K_MAX for the
same key with n_confident >= 1 ("loud nearby").

Outputs:
  f            : fraction of silent instance-frames loud within ±K (K sweep)
  voxel mass   : FN voxels recoverable if every loud-nearby silent instance
                 were fully claimed (oracle, zero added FP)
  oracle IoU   : (TP + recovered) / (TP + FP + FN)  <- the Phase-3 ceiling
  |k| histogram to nearest loud frame  <- sets the propagation window

PRE-REGISTERED DECISION:
  oracle IoU <= 0.66  -> KILL. Temporal consistency can't clear the 0.6430
                         bar meaningfully; project result is terminal and the
                         paper gains a third information-limitation result.
  oracle IoU >= 0.70  -> PROCEED to Phase-3 propagation design (still gated:
                         realized gain will be below oracle; design must
                         state its FP budget up front).
  between             -> judgment call; look at the |k| histogram and the
                         precision headroom (P=0.907) before deciding.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/gate_temporal_oracle.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml \
    --ckpt ~/sparseunet4d/runs/residual_v2/best.pt \
    --cache ~/sparseunet4d/runs/residual_v2/temporal_oracle.npz
  # re-analyze without GPU:
  python3 gate_temporal_oracle.py --analyze-only --cache .../temporal_oracle.npz
"""
import os, sys, argparse, yaml
import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))

TH = 0.5
K_SWEEP = [1, 2, 3, 5, 10, 20]
K_MAX = max(K_SWEEP)


# ---------------------------------------------------------------------------
# pass 1: GPU sweep -> per-instance table
# ---------------------------------------------------------------------------
def build_cache(args):
    import torch
    from torch.utils.data import DataLoader
    from sparseunet4d.datasets import SemanticKITTI4D, me_collate
    from sparseunet4d.datasets.semantickitti import _read_scan, _read_label
    from sparseunet4d.datasets.label_map import split_label, MOVING_IDS
    from sparseunet4d.models.backend import backend
    from sparseunet4d.models.model import SparseUNet4D

    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vox = d["voxel_size"]; prange = d["point_range"]

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        vox, d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], prange,
        residual_feats=d.get("residual_feats", True),
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

    rows = []                       # (frame, inst_key, n_vox, n_conf)
    tp = fp = fn = 0
    root = d["root"]
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            seq, ref = batch["meta"][0]
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
            cnp = batch["coords"].numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup] >= TH
            tp += int((pm & (g == 1)).sum()); fp += int((pm & (g == 0)).sum())
            fn += int((~pm & (g == 1)).sum())

            # --- instance keys for GT-moving voxels ------------------------
            # rebuild reference-frame point->voxel keys with the SAME pipeline
            seq_dir = os.path.join(root, f"{seq:02d}")
            scan = _read_scan(os.path.join(seq_dir, "velodyne", f"{ref:06d}.bin"))
            xyz = scan[:, :3]
            lab = _read_label(os.path.join(seq_dir, "labels", f"{ref:06d}.label"))
            sem_raw, inst_raw = split_label(lab)
            mask = np.all(np.abs(xyz) < prange, axis=1)
            xyz, sem_raw, inst_raw = xyz[mask], sem_raw[mask], inst_raw[mask]
            mov = np.isin(sem_raw, list(MOVING_IDS))
            if not mov.any():
                if bi % 400 == 0: print(f"  frame {bi}/{len(loader)}", flush=True)
                continue
            vk = np.floor(xyz[mov] / vox).astype(np.int64)          # (P,3)
            ikey = sem_raw[mov] * 100000 + inst_raw[mov]

            # voxel prob lookup: supervised t==0 voxels from model output
            sup_c = cnp[sup]                                        # (M,5)
            t0 = sup_c[:, 4] == 0
            vox_xyz = sup_c[t0][:, 1:4].astype(np.int64)
            vox_prob = prob[sup][t0]
            # dict via structured view
            def keyize(a):
                return (a[:, 0].astype(np.int64) * 2**42
                        + (a[:, 1] + 2**20) * 2**21 + (a[:, 2] + 2**20))
            lut = dict(zip(keyize(vox_xyz).tolist(), vox_prob.tolist()))
            pprob = np.array([lut.get(k, 0.0) for k in keyize(vk).tolist()],
                             np.float32)

            # per (instance, voxel) dedup -> per-instance counts
            uk, first = np.unique(
                np.stack([ikey, keyize(vk)], 1), axis=0, return_index=True)
            inst_of = uk[:, 0]; conf = pprob[first] >= TH
            for k in np.unique(inst_of):
                s = inst_of == k
                rows.append((ref, int(k), int(s.sum()), int(conf[s].sum())))
            if bi % 400 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    arr = np.array(rows, np.int64)
    np.savez_compressed(args.cache, table=arr, tp=tp, fp=fp, fn=fn)
    print(f"cached {len(arr)} instance-frames -> {args.cache}")
    print(f"baseline voxel IoU sanity: "
          f"{tp / max(tp + fp + fn, 1):.4f} (expect ~0.6235)")


# ---------------------------------------------------------------------------
# pass 2: CPU analysis
# ---------------------------------------------------------------------------
def analyze(args):
    z = np.load(args.cache)
    T = z["table"]; tp, fp, fn = int(z["tp"]), int(z["fp"]), int(z["fn"])
    frames, keys, nvox, nconf = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    print(f"instance-frames: {len(T)}  unique instances: {len(np.unique(keys))}")
    print(f"baseline: IoU={tp/max(tp+fp+fn,1):.4f} TP={tp} FP={fp} FN={fn}")

    # loud lookup: set of (key, frame) with nconf >= 1
    loud = {}
    for k, f in zip(keys[nconf >= 1], frames[nconf >= 1]):
        loud.setdefault(k, []).append(f)
    loud = {k: np.sort(np.array(v)) for k, v in loud.items()}

    silent = nconf == 0
    print(f"\nsilent instance-frames: {silent.sum()} "
          f"({nvox[silent].sum()} FN voxels; "
          f"{100 * nvox[silent].sum() / max(fn, 1):.0f}% of all FN)")

    nearest = np.full(silent.sum(), 10**9)
    sk, sf, sv = keys[silent], frames[silent], nvox[silent]
    for i, (k, f) in enumerate(zip(sk, sf)):
        lf = loud.get(k)
        if lf is not None and len(lf):
            nearest[i] = np.abs(lf - f).min()

    print(f"\n{'K':>4} {'f (loud<=K)':>12} {'recov vox':>10} {'oracle IoU':>11}")
    for K in K_SWEEP:
        hit = nearest <= K
        rec = int(sv[hit].sum())
        oiou = (tp + rec) / max(tp + fp + fn, 1)
        print(f"{K:>4} {hit.mean():>12.3f} {rec:>10d} {oiou:>11.4f}")

    ever = nearest < 10**9
    rec = int(sv[ever].sum())
    print(f" ever {ever.mean():>12.3f} {rec:>10d} "
          f"{(tp + rec) / max(tp + fp + fn, 1):>11.4f}")

    h, _ = np.histogram(np.clip(nearest[ever], 0, 30), bins=range(32))
    print("\n|k| to nearest loud frame (capped 30):")
    print("  " + " ".join(f"{i}:{c}" for i, c in enumerate(h) if c))

    best_k10 = (tp + int(sv[nearest <= 10].sum())) / max(tp + fp + fn, 1)
    verdict = ("PROCEED to Phase-3 design" if best_k10 >= 0.70 else
               "KILL -- temporal consistency cannot clear the bar"
               if best_k10 <= 0.66 else "GREY ZONE -- inspect histogram/P headroom")
    print(f"\noracle IoU @ K=10 = {best_k10:.4f}  ->  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        assert args.config and args.ckpt, "--config and --ckpt required for pass 1"
        build_cache(args)
    analyze(args)