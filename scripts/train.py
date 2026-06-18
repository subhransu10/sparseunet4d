"""Training loop for SparseUNet4D (joint MOS + semantic).

Backend-agnostic: CPU mock for smoke tests, MinkowskiEngine for real training
(SU4D_BACKEND=me). Optional drift forward pass drives the consistency loss.

Features:
  - Checkpointing: last.pt every val_every iters; best.pt on best val moving-IoU.
  - Resume: --resume path.pt restores model + optimizer + iter + best-IoU.
  - Configurable dataloader workers (train.num_workers) for speed.
  - Early stop: stop if val moving-IoU hasn't improved for `patience` validations.
  - Timing: reports data-loading vs compute time so you can find the bottleneck.

Usage:
  SU4D_BACKEND=me python scripts/train.py --config configs/semantickitti_base.yaml \
      --save-dir runs/full
  # resume:
  SU4D_BACKEND=me python scripts/train.py --config ... --save-dir runs/full \
      --resume runs/full/last.pt
"""
from __future__ import annotations
import os, sys, time, argparse, yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.models.losses import total_loss
from sparseunet4d.utils.metrics import IoUMeter


def to_st(batch, device):
    coords = batch["coords"].to(device)
    feats = batch["feats"].to(device)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats, coordinates=coords)
    return ST(feats, coords)


def run_batch(model, batch, cfg, device, drift_batch=None):
    x = to_st(batch, device)
    out = model(x)
    out_drift = None
    if drift_batch is not None and cfg["loss"].get("consistency_weight", 0) > 0:
        out_drift = model(to_st(drift_batch, device))
    loss, parts = total_loss(out, batch["motion"].to(device),
                             batch["semantic"].to(device), cfg["loss"], out_drift)
    return out, loss, parts


def validate(model, loader, cfg, device, num_sem):
    model.eval()
    mos, sem = IoUMeter(2), IoUMeter(num_sem)
    with torch.no_grad():
        for batch in loader:
            out, _, _ = run_batch(model, batch, cfg, device)
            mos.update(out["motion_logits"], batch["motion"].to(device))
            sem.update(out["semantic_logits"], batch["semantic"].to(device))
    model.train()
    return {"moving_iou": mos.moving_iou(), "semantic_miou": sem.miou()}


def save_ckpt(path, model, opt, cfg, it, best_iou, no_improve):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "iter": it, "best_moving_iou": best_iou,
                "no_improve": no_improve, "config": cfg}, path)


def train(cfg, train_loader, val_loader=None, device="cpu", drift_loader=None,
          max_iters=None, log_every=50, save_dir="runs/exp", resume=None):
    os.makedirs(save_dir, exist_ok=True)
    num_sem = cfg["dataset"].get("num_semantic", 20)
    model = SparseUNet4D(
        in_ch=1, num_semantic=num_sem,
        use_se=cfg["model"].get("use_se", True),
        use_ego_decouple=cfg["model"].get("use_ego_decouple", True),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    max_iters = max_iters or cfg["train"]["iters"]
    val_every = cfg["train"].get("val_every", 1000)
    patience = cfg["train"].get("patience", 0)        # 0 = disabled
    drift_iter = iter(drift_loader) if drift_loader is not None else None

    it, best_iou, no_improve = 0, -1.0, 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        it = ck.get("iter", 0)
        best_iou = ck.get("best_moving_iou", -1.0)
        no_improve = ck.get("no_improve", 0)
        print(f"resumed from {resume} @ iter {it} (best={best_iou:.4f})")

    def checkpoint_and_eval(it):
        nonlocal best_iou, no_improve
        save_ckpt(os.path.join(save_dir, "last.pt"), model, opt, cfg, it,
                  best_iou, no_improve)
        if val_loader is None:
            return False
        m = validate(model, val_loader, cfg, device, num_sem)
        print(f"  [val @ {it}] moving_iou={m['moving_iou']:.4f} "
              f"semantic_miou={m['semantic_miou']:.4f}", flush=True)
        if m["moving_iou"] > best_iou:
            best_iou = m["moving_iou"]; no_improve = 0
            save_ckpt(os.path.join(save_dir, "best.pt"), model, opt, cfg, it,
                      best_iou, no_improve)
            print(f"  [val @ {it}] new best -> best.pt ({best_iou:.4f})", flush=True)
        else:
            no_improve += 1
        if patience and no_improve >= patience:
            print(f"early stop: no improvement for {patience} validations "
                  f"(best={best_iou:.4f})", flush=True)
            return True
        return False

    t_data = t_compute = 0.0
    model.train()
    stop = False
    while it < max_iters and not stop:
        t0 = time.time()
        for batch in train_loader:
            t_data += time.time() - t0
            drift_batch = None
            if drift_iter is not None:
                try:
                    drift_batch = next(drift_iter)
                except StopIteration:
                    drift_iter = iter(drift_loader)
                    drift_batch = next(drift_iter)
            tc = time.time()
            opt.zero_grad()
            _, loss, parts = run_batch(model, batch, cfg, device, drift_batch)
            loss.backward()
            opt.step()
            t_compute += time.time() - tc
            if it % log_every == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(f"[iter {it}] loss={loss.item():.3f} {msg} "
                      f"| data {t_data:.1f}s compute {t_compute:.1f}s", flush=True)
                t_data = t_compute = 0.0
            it += 1
            if it % val_every == 0 or it >= max_iters:
                if checkpoint_and_eval(it):
                    stop = True; break
            if it >= max_iters:
                break
            t0 = time.time()
    print(f"done @ iter {it}. best moving_iou={best_iou:.4f}  ckpts in {save_dir}")
    return model


def _load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--save-dir", default="runs/exp")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    from sparseunet4d.datasets import SemanticKITTI4D, me_collate
    d = cfg["dataset"]; p = cfg["pose"]
    nw = cfg["train"].get("num_workers", 8)
    train_ds = SemanticKITTI4D(
        d["root"], d["train_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], p["mode"], p["rot_std_deg"], p["trans_std_m"], p["seed"],
        d["point_range"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=me_collate, num_workers=nw,
                              persistent_workers=(nw > 0), pin_memory=True)
    val_ds = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, collate_fn=me_collate, num_workers=nw,
                            persistent_workers=(nw > 0), pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(cfg, train_loader, val_loader=val_loader, device=device,
          max_iters=args.iters, save_dir=args.save_dir, resume=args.resume)