"""
One-batch forward+loss smoke test for the residual-channel (in_ch=4) path.
Runs the REAL pipeline: dataset -> me_collate -> to_st -> model -> total_loss
-> backward. Confirms 4-channel feats flow through the stem and grads are finite.

  3050 / CPU-mock :  python smoke_test.py
  5090 / real ME  :  SU4D_BACKEND=me python smoke_test.py   (few seconds, not a GPU-day)
"""
import os, sys, argparse, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D
from sparseunet4d.models.losses import total_loss
from sparseunet4d.datasets import SemanticKITTI4D, me_collate

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/semantickitti_base.yaml")
ap.add_argument("--seq", type=int, default=8)
ap.add_argument("--idx", type=int, nargs=2, default=[200, 800])
args = ap.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
d, num_sem = cfg["dataset"], cfg["dataset"].get("num_semantic", 20)
n_frames = d.get("n_frames", 4)
residual_feats = d.get("residual_feats", True)
in_ch = 1 + (n_frames - 1) if residual_feats else 1
device = "cuda" if (backend() == "me" and torch.cuda.is_available()) else "cpu"
print(f"backend={backend()}  device={device}  in_ch={in_ch}  n_frames={n_frames}")

ds = SemanticKITTI4D(d["root"], [args.seq], n_frames, d["voxel_size"],
                     d["semantic_yaml"], "gt", 0.0, 0.0, 0, d["point_range"],
                     residual_feats=residual_feats, res_clip=d.get("res_clip", 3.0))
batch = me_collate([ds[args.idx[0]], ds[args.idx[1]]])
print(f"batch feats {tuple(batch['feats'].shape)}  coords {tuple(batch['coords'].shape)}")
assert batch["feats"].shape[1] == in_ch, "feat width != in_ch"

mc = cfg.get("model", {})
model = SparseUNet4D(in_ch=in_ch, num_semantic=num_sem, base=mc.get("base", 32),
                     n_stages=mc.get("n_stages", 2),
                     use_se=mc.get("use_se", True)).to(device)

def to_st(b):
    coords, feats = b["coords"].to(device), b["feats"].to(device)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats, coordinates=coords)
    return ST(feats, coords)

model.train()
out = model(to_st(batch))
loss, parts = total_loss(out, batch["motion"].to(device),
                         batch["semantic"].to(device), cfg["loss"], None)
print(f"forward OK  motion_logits {tuple(out['motion_logits'].shape)}  "
      f"loss={loss.item():.4f}  parts=" + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
assert torch.isfinite(loss), "loss is not finite"

loss.backward()
gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
print(f"backward OK  params_with_grad={n_grad}  total_grad_norm={gnorm:.4f}")
assert gnorm > 0 and gnorm == gnorm, "zero or NaN gradients"

print("\nSMOKE TEST PASSED -> launch the 5k sanity run.")