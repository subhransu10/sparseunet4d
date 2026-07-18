"""Phase-2 drift-consistency fine-tuning (the ego-motion-robustness contribution).

Fine-tunes a trained checkpoint so its motion prediction is INVARIANT to
realistic odometry drift, instead of trusting registration. Each step runs the
SAME sample twice — once with GT poses, once through the compounding-drift
provider — and adds a symmetric-KL consistency term between the two motion
distributions on the reference voxels, plus (optionally) direct supervision of
the drifted pass.

Why the pairing is sound: pose registration only moves PAST frames; the
reference frame (t=0) is untouched. So both passes share an identical t=0
voxel set, and within a sample np.unique's lexicographic order makes the
supervised rows (motion label != IGNORE) align 1:1. We assert this on every
batch rather than trusting it.

The prior `drift_loader` hook in train.py paired *different* samples (KL
between different scenes — meaningless); this trainer replaces that idea.

Usage:
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/scripts/train_consistency.py \
    --config ~/sparseunet4d/configs/consistency_ft.yaml \
    --init ~/sparseunet4d/runs/residual_temporal/best.pt \
    --save-dir ~/sparseunet4d/runs/consistency_ft
"""
from __future__ import annotations
import os, sys, math, time, argparse, yaml
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.models.losses import total_loss, consistency_loss
from sparseunet4d.utils.metrics import IoUMeter, MovingThresholdMeter


class PairedDriftDataset(Dataset):
    """Same index -> (clean-pose sample, drifted-pose sample). No random
    augmentation/injection in either view, so alignment is deterministic."""

    def __init__(self, d, p_drift):
        common = dict(
            root=d["root"], sequences=d["train_sequences"],
            n_frames=d["n_frames"], voxel_size=d["voxel_size"],
            semantic_yaml=d["semantic_yaml"], point_range=d["point_range"],
            residual_feats=d.get("residual_feats", True),
            res_clip=d.get("res_clip", 3.0),
            frame_offsets=d.get("frame_offsets"))
        self.clean = SemanticKITTI4D(pose_mode="gt", rot_std_deg=0.0,
                                     trans_std_m=0.0, pose_seed=0, **common)
        self.drift = SemanticKITTI4D(pose_mode="drift",
                                     rot_std_deg=p_drift["rot_std_deg"],
                                     trans_std_m=p_drift["trans_std_m"],
                                     pose_seed=p_drift.get("seed", 0), **common)
        assert len(self.clean) == len(self.drift)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return {"clean": self.clean[i], "drift": self.drift[i]}


def paired_collate(batch):
    return {"clean": me_collate([b["clean"] for b in batch]),
            "drift": me_collate([b["drift"] for b in batch])}


def to_st(b, dev):
    coords = b["coords"].to(dev); feats = b["feats"].to(dev)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats, coordinates=coords)
    return ST(feats, coords)


def masked_consistency(out_c, out_d, batch_c, batch_d):
    """Symmetric KL on the aligned reference voxels of the two passes."""
    mc = batch_c["motion"] != -1
    md = batch_d["motion"] != -1
    cc = batch_c["coords"][mc]      # [b, x, y, z, t] of supervised voxels
    cd = batch_d["coords"][md]
    if cc.shape != cd.shape or not torch.equal(cc, cd):
        raise RuntimeError("clean/drift reference voxels misaligned -- "
                           "was augmentation or injection enabled?")
    return consistency_loss(out_c["motion_logits"][mc.to(out_c["motion_logits"].device)],
                            out_d["motion_logits"][md.to(out_d["motion_logits"].device)])


def validate(model, loader, dev, num_sem):
    model.eval(); mos, sem = MovingThresholdMeter(), IoUMeter(num_sem)
    with torch.no_grad():
        for batch in loader:
            out = model(to_st(batch, dev))
            mos.update(out["motion_logits"], batch["motion"].to(dev))
            sem.update(out["semantic_logits"], batch["semantic"].to(dev))
    model.train()
    b = mos.best()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", required=True, help="checkpoint to fine-tune")
    ap.add_argument("--save-dir", default="runs/consistency_ft")
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; m = cfg["model"]; t = cfg["train"]
    pd = cfg["pose_drift"]; L = cfg["loss"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    num_sem = d.get("num_semantic", 20)

    pair_ds = PairedDriftDataset(d, pd)
    nw = t.get("num_workers", 8)
    loader = DataLoader(pair_ds, batch_size=t["batch_size"], shuffle=True,
                        collate_fn=paired_collate, num_workers=nw,
                        persistent_workers=(nw > 0), pin_memory=True)
    # validation stays CLEAN-pose: robustness is measured separately by the
    # drift sweep; selecting on clean IoU guards against sacrificing accuracy.
    val_ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, 0,
        d["point_range"], residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"))
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, num_sem, base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True)).to(dev)
    ck = torch.load(args.init, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    print(f"initialized from {args.init} "
          f"(best_moving_iou={ck.get('best_moving_iou')})", flush=True)

    base_lr = t["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=t["weight_decay"])
    max_iters = args.iters or t["iters"]
    warmup = t.get("warmup_iters", 300)
    val_every = t.get("val_every", 500)
    cw = L.get("consistency_weight", 1.0)
    dsw = L.get("drift_sup_weight", 1.0)

    it, best_iou = 0, -1.0
    model.train()
    while it < max_iters:
        for pair in loader:
            for g in opt.param_groups:
                prog = max(0.0, (it - warmup) / max(1, max_iters - warmup))
                g["lr"] = (base_lr * (it + 1) / warmup if it < warmup
                           else base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))
            bc, bd = pair["clean"], pair["drift"]
            out_c = model(to_st(bc, dev))
            out_d = model(to_st(bd, dev))
            l_c, parts = total_loss(out_c, bc["motion"].to(dev),
                                    bc["semantic"].to(dev), L)
            l_d, _ = total_loss(out_d, bd["motion"].to(dev),
                                bd["semantic"].to(dev), L)
            l_con = masked_consistency(out_c, out_d, bc, bd)
            loss = l_c + dsw * l_d + cw * l_con
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f"[iter {it}] lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={loss.item():.3f} clean={l_c.item():.3f} "
                      f"drift={l_d.item():.3f} consist={l_con.item():.4f}",
                      flush=True)
            it += 1
            if it % val_every == 0 or it >= max_iters:
                v = validate(model, val_loader, dev, num_sem)
                print(f"  [val @ {it}] moving_iou={v['iou']:.4f}@th{v['threshold']:.2f} "
                      f"P={v['prec']:.3f} R={v['rec']:.3f}", flush=True)
                torch.save({"model": model.state_dict(), "iter": it,
                            "best_moving_iou": max(best_iou, v["iou"]),
                            "best_threshold": v["threshold"], "config": cfg},
                           os.path.join(args.save_dir, "last.pt"))
                if v["iou"] > best_iou:
                    best_iou = v["iou"]
                    torch.save({"model": model.state_dict(), "iter": it,
                                "best_moving_iou": best_iou,
                                "best_threshold": v["threshold"], "config": cfg},
                               os.path.join(args.save_dir, "best.pt"))
                    print(f"  [val @ {it}] new best -> best.pt ({best_iou:.4f})",
                          flush=True)
            if it >= max_iters:
                break
    print(f"done @ {it}. best clean moving_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
