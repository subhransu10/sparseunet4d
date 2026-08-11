"""Training loop for SparseUNet4D (joint MOS + semantic).

Adds LR warmup + cosine decay and gradient accumulation to stabilise larger
models (fixes the val oscillation seen at base=64/3-stage, batch=2).

Config knobs (train:):
  lr, weight_decay, iters, val_every, num_workers, patience
  warmup_iters   : linear warmup steps (default 500)
  accum_steps    : gradient-accumulation micro-batches (default 1 = off)
  lr_schedule    : 'cosine' (default) or 'none'

Usage:
  SU4D_BACKEND=me python scripts/train.py --config configs/semantickitti_big.yaml \
      --save-dir runs/big [--resume runs/big/last.pt]
"""
from __future__ import annotations
import os, sys, math, time, argparse, yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.models.losses import total_loss
from sparseunet4d.utils.metrics import IoUMeter, MovingThresholdMeter



def build_model(cfg_model, in_ch, num_sem):
    """SparseUNet4D, or the decoupled motion/appearance DualBranchUNet4D when
    model.dual_branch is set. Same heads + cluster head either way."""
    m = cfg_model
    common = dict(use_cluster=m.get("use_cluster", False),
                  cluster_link=m.get("cluster_link", 2),
                  cluster_min_size=m.get("cluster_min_size", 3),
                  cluster_cross_frame=m.get("cluster_cross_frame", False),
                  cluster_feature_fusion=m.get("cluster_feature_fusion", False),
                  cluster_fg_mode=m.get("cluster_fg_mode", "semantic"),
                  cluster_fg_seed_th=m.get("cluster_fg_seed_th", 0.3))
    if m.get("dual_branch", False):
        from sparseunet4d.models.dual_branch import DualBranchUNet4D
        return DualBranchUNet4D(in_ch, num_sem, base=m.get("base", 32),
                                n_stages=m.get("n_stages", 2),
                                use_se=m.get("use_se", True),
                                app_ch=m.get("app_ch", 1), **common)
    return SparseUNet4D(in_ch, num_sem, base=m.get("base", 32),
                        n_stages=m.get("n_stages", 2),
                        use_se=m.get("use_se", True),
                        use_ego_decouple=m.get("use_ego_decouple", False),
                        **common)


def to_st(batch, device):
    coords = batch["coords"].to(device); feats = batch["feats"].to(device)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats, coordinates=coords)
    return ST(feats, coords)


def run_batch(model, batch, cfg, device, drift_batch=None):
    x = to_st(batch, device); out = model(x); out_drift = None
    if drift_batch is not None and cfg["loss"].get("consistency_weight", 0) > 0:
        out_drift = model(to_st(drift_batch, device))
    off = batch.get("offset"); offm = batch.get("offset_mask")
    loss, parts = total_loss(out, batch["motion"].to(device),
                             batch["semantic"].to(device), cfg["loss"], out_drift,
                             offset_gt=off.to(device) if off is not None else None,
                             offset_mask=offm.to(device) if offm is not None else None,
                             motion_instance=(batch["motion_instance"].to(device)
                                              if "motion_instance" in batch else None),
                             point_count=(batch["point_count"].to(device)
                                          if "point_count" in batch else None))
    return out, loss, parts


def validate(model, loader, cfg, device, num_sem):
    thresholds = cfg["train"].get("checkpoint_thresholds")
    model.eval()
    mos_voxel = MovingThresholdMeter(thresholds)
    mos_point = MovingThresholdMeter(thresholds)
    sem = IoUMeter(num_sem)
    have_points = False
    with torch.no_grad():
        for batch in loader:
            out, _, _ = run_batch(model, batch, cfg, device)
            mos_voxel.update(out["motion_logits"], batch["motion"].to(device))
            if "ref_point_voxel" in batch:
                rows = batch["ref_point_voxel"].to(device)
                mos_point.update(out["motion_logits"][rows],
                                 batch["ref_point_motion"].to(device))
                have_points = True
            sem.update(out["semantic_logits"], batch["semantic"].to(device))
    model.train()
    voxel = mos_voxel.best()
    point = mos_point.best() if have_points else None
    metric = cfg["train"].get("checkpoint_metric", "voxel")
    if metric not in ("voxel", "point"):
        raise ValueError("train.checkpoint_metric must be 'voxel' or 'point'")
    if metric == "point" and point is None:
        raise ValueError("point checkpoint metric requires return_point_map=True")
    b = point if metric == "point" else voxel
    # 'moving_iou' is now the threshold-optimal IoU (used for model selection);
    # 'moving_iou_argmax' keeps the old argmax@0.5 number for comparison.
    return {"moving_iou": b["iou"], "moving_iou_argmax": b["iou_argmax"],
            "moving_threshold": b["threshold"], "semantic_miou": sem.miou(),
            "moving_prec": b["prec"], "moving_rec": b["rec"],
            "checkpoint_metric": metric,
            "voxel_iou": voxel["iou"],
            "point_iou": point["iou"] if point is not None else None}


def save_ckpt(path, model, opt, cfg, it, best_iou, no_improve, best_threshold=0.5):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "iter": it, "best_moving_iou": best_iou,
                "best_threshold": best_threshold,
                "no_improve": no_improve, "config": cfg}, path)


def lr_at(it, base_lr, warmup, total, schedule):
    if warmup and it < warmup:
        return base_lr * (it + 1) / warmup
    if schedule == "cosine":
        prog = (it - warmup) / max(1, total - warmup)
        return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return base_lr


def train(cfg, train_loader, val_loader=None, device="cpu", drift_loader=None,
          max_iters=None, log_every=50, save_dir="runs/exp", resume=None,
          init=None):
    os.makedirs(save_dir, exist_ok=True)
    num_sem = cfg["dataset"].get("num_semantic", 20)
    m = cfg.get("model", {})
    d = cfg.get("dataset", {})
    n_frames = d.get("n_frames", 4)                     # d = dataset config section
    residual_feats = d.get("residual_feats", True)
    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)
    in_ch = 1 + _k if residual_feats else 1
    model = build_model(m, in_ch, num_sem).to(device)
    base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=cfg["train"]["weight_decay"])
    max_iters = max_iters or cfg["train"]["iters"]
    val_every = cfg["train"].get("val_every", 1000)
    patience = cfg["train"].get("patience", 0)
    warmup = cfg["train"].get("warmup_iters", 500)
    accum = max(1, cfg["train"].get("accum_steps", 1))
    schedule = cfg["train"].get("lr_schedule", "cosine")
    drift_iter = iter(drift_loader) if drift_loader is not None else None

    it, best_iou, no_improve, best_thr = 0, -1.0, 0, 0.5
    if resume and init:
        raise ValueError("use either resume or init, not both")
    if init:
        ck = torch.load(init, map_location=device)
        state = ck["model"] if "model" in ck else ck
        model.load_state_dict(state, strict=True)
        print(f"initialized model weights from {init}")
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        it = ck.get("iter", 0); best_iou = ck.get("best_moving_iou", -1.0)
        no_improve = ck.get("no_improve", 0); best_thr = ck.get("best_threshold", 0.5)
        print(f"resumed from {resume} @ iter {it} (best={best_iou:.4f})")

    def checkpoint_and_eval(it):
        nonlocal best_iou, no_improve, best_thr
        save_ckpt(os.path.join(save_dir, "last.pt"), model, opt, cfg, it, best_iou,
                  no_improve, best_thr)
        if val_loader is None: return False
        v = validate(model, val_loader, cfg, device, num_sem)
        print(f"  [val @ {it}] moving_iou={v['moving_iou']:.4f}@th{v['moving_threshold']:.2f} "
              f"[{v['checkpoint_metric']}] "
              f"(argmax {v['moving_iou_argmax']:.4f}) "
              f"P={v['moving_prec']:.3f} R={v['moving_rec']:.3f} "
              f"voxel={v['voxel_iou']:.4f} "
              f"point={v['point_iou'] if v['point_iou'] is not None else float('nan'):.4f} "
              f"semantic_miou={v['semantic_miou']:.4f}", flush=True)
        if v["moving_iou"] > best_iou:
            best_iou = v["moving_iou"]; no_improve = 0; best_thr = v["moving_threshold"]
            save_ckpt(os.path.join(save_dir, "best.pt"), model, opt, cfg, it, best_iou,
                      no_improve, best_thr)
            print(f"  [val @ {it}] new best -> best.pt ({best_iou:.4f} @ th {best_thr:.2f})",
                  flush=True)
        else:
            no_improve += 1
        if patience and no_improve >= patience:
            print(f"early stop: no improvement for {patience} validations "
                  f"(best={best_iou:.4f})", flush=True)
            return True
        return False

    t_data = t_compute = 0.0
    model.train(); opt.zero_grad(); stop = False
    while it < max_iters and not stop:
        t0 = time.time()
        for batch in train_loader:
            t_data += time.time() - t0
            drift_batch = None
            if drift_iter is not None:
                try: drift_batch = next(drift_iter)
                except StopIteration:
                    drift_iter = iter(drift_loader); drift_batch = next(drift_iter)
            for g in opt.param_groups:
                g["lr"] = lr_at(it, base_lr, warmup, max_iters, schedule)
            tc = time.time()
            _, loss, parts = run_batch(model, batch, cfg, device, drift_batch)
            (loss / accum).backward()
            if (it + 1) % accum == 0:
                opt.step(); opt.zero_grad()
            t_compute += time.time() - tc
            if it % log_every == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(f"[iter {it}] lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={loss.item():.3f} {msg} "
                      f"| data {t_data:.1f}s compute {t_compute:.1f}s", flush=True)
                t_data = t_compute = 0.0
            it += 1
            if it % val_every == 0 or it >= max_iters:
                if checkpoint_and_eval(it): stop = True; break
            if it >= max_iters: break
            t0 = time.time()
    print(f"done @ iter {it}. best moving_iou={best_iou:.4f}  ckpts in {save_dir}")
    return model


def _load_cfg(path):
    with open(path) as f: cfg = yaml.safe_load(f)
    cfg.setdefault("model", {}); return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--save-dir", default="runs/exp")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None,
                    help="warm-start model weights only; optimizer/schedule restart")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    from sparseunet4d.datasets import SemanticKITTI4D, me_collate
    d = cfg["dataset"]; p = cfg["pose"]; nw = cfg["train"].get("num_workers", 8)
    train_ds = SemanticKITTI4D(d["root"], d["train_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], p["mode"], p["rot_std_deg"],
        p["trans_std_m"], p["seed"], d["point_range"],
        residual_feats=d.get("residual_feats", True), res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"), augment=d.get("augment", False),
        inject_bank=d.get("inject_bank"), inject_prob=d.get("inject_prob", 0.0),
        inject_max_n=d.get("inject_max_n", 4), feat_rep=d.get("feat_rep", "label"),
        all_frame_labels=d.get("all_frame_labels", False),   # P3: train only
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        inject_class_boost=d.get("inject_class_boost"),
        inject_all_frame_labels=d.get("inject_all_frame_labels", False))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, collate_fn=me_collate, num_workers=nw,
        persistent_workers=(nw > 0), pin_memory=True)
    val_ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"], d["point_range"],
        residual_feats=d.get("residual_feats", True), res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        return_point_map=(cfg["train"].get("checkpoint_metric", "voxel") == "point"))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, collate_fn=me_collate, num_workers=nw,
        persistent_workers=(nw > 0), pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(cfg, train_loader, val_loader=val_loader, device=device,
          max_iters=args.iters, save_dir=args.save_dir, resume=args.resume,
          init=args.init)
