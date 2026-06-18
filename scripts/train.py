"""Training loop for SparseUNet4D (joint MOS + semantic).

Backend-agnostic: runs on the CPU mock for smoke tests and on MinkowskiEngine
for real training (SU4D_BACKEND=me). Optionally does a second forward pass on a
drift-perturbed copy of the batch to drive the consistency loss (Phase 2).

Usage:
  SU4D_BACKEND=me python scripts/train.py --config configs/semantickitti_base.yaml
"""
from __future__ import annotations
import os, sys, argparse, yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.models.losses import total_loss
from sparseunet4d.utils.metrics import IoUMeter


def to_st(batch, device):
    """Build the backend sparse tensor from a collated batch."""
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


def train(cfg, train_loader, val_loader=None, device="cpu",
          drift_loader=None, max_iters=None, log_every=50):
    num_sem = cfg["dataset"].get("num_semantic", 20)
    model = SparseUNet4D(
        in_ch=1, num_semantic=num_sem,
        use_se=cfg["model"].get("use_se", True),
        use_ego_decouple=cfg["model"].get("use_ego_decouple", True),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    max_iters = max_iters or cfg["train"]["iters"]
    drift_iter = iter(drift_loader) if drift_loader is not None else None

    it = 0
    model.train()
    while it < max_iters:
        for batch in train_loader:
            drift_batch = None
            if drift_iter is not None:
                try:
                    drift_batch = next(drift_iter)
                except StopIteration:
                    drift_iter = iter(drift_loader)
                    drift_batch = next(drift_iter)
            opt.zero_grad()
            _, loss, parts = run_batch(model, batch, cfg, device, drift_batch)
            loss.backward()
            opt.step()
            if it % log_every == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(f"[iter {it}] loss={loss.item():.3f} {msg}")
            it += 1
            if it >= max_iters:
                break
        if val_loader is not None and cfg["train"].get("val_every"):
            print("  val:", validate(model, val_loader, cfg, device, num_sem))
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
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    from sparseunet4d.datasets import SemanticKITTI4D, me_collate
    d = cfg["dataset"]; p = cfg["pose"]
    train_ds = SemanticKITTI4D(
        d["root"], d["train_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], p["mode"], p["rot_std_deg"], p["trans_std_m"], p["seed"],
        d["point_range"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=me_collate, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(cfg, train_loader, device=device, max_iters=args.iters)
