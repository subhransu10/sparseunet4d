"""Model logic tests on the CPU mock backend (runs on the 3050 / any CPU).

  1. Full forward over a tiny 4D batch -> correct output shapes.
  2. Backward -> gradients reach the ego-decoupling module and the heads.
  3. EgoDecoupleBlock removes a *purely global* apparent shift (residual ~ 0),
     while preserving a *local* one -> the mechanism does what it claims.

Run:  SU4D_BACKEND=mock python tools/test_model.py
"""
import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SU4D_BACKEND", "mock")
from sparseunet4d.models.backend import ST, temporal_difference, global_avg_pool_per_batch, masked_mean_per_batch
from sparseunet4d.models.model import SparseUNet4D, EgoDecoupleBlock
from sparseunet4d.models.losses import total_loss


def _toy_batch(n_per_frame=150, n_frames=3, B=2, ext=64):
    """Random occupied voxels across B scenes and n_frames temporal slices."""
    coords, feats = [], []
    for b in range(B):
        for t in range(n_frames):
            xyz = torch.randint(0, ext, (n_per_frame, 3))
            c = torch.cat([torch.full((n_per_frame, 1), b),
                           xyz,
                           torch.full((n_per_frame, 1), t)], dim=1)
            coords.append(c)
            feats.append(torch.randn(n_per_frame, 1))
    coords = torch.cat(coords).long()
    coords = torch.unique(coords, dim=0)            # dedup voxels
    feats = torch.randn(coords.shape[0], 1)
    return ST(feats, coords)


def test_forward_backward():
    x = _toy_batch()
    model = SparseUNet4D(in_ch=1, num_semantic=20, base=16)
    out = model(x)
    N = out["motion_logits"].shape[0]
    assert out["motion_logits"].shape == (N, 2)
    assert out["semantic_logits"].shape == (N, 20)
    print(f"  [ok] forward: {x.coords.shape[0]} in -> {N} out voxels, "
          f"motion {tuple(out['motion_logits'].shape)}, "
          f"semantic {tuple(out['semantic_logits'].shape)}")

    # supervise reference voxels (t==0); ignore the rest
    t = out["coords"][:, 4]
    mot = torch.where(t == 0, torch.randint(0, 2, (N,)), torch.full((N,), -1))
    sem = torch.where(t == 0, torch.randint(0, 20, (N,)), torch.full((N,), -1))
    cfg = {"moving_class_weight": 5.0, "dice_weight": 1.0}
    loss, parts = total_loss(out, mot, sem, cfg)
    loss.backward()

    ego_grad = model.ego.gate[0].weight.grad
    head_grad = model.motion_head.weight.grad
    assert ego_grad is not None and ego_grad.abs().sum() > 0, "no grad to ego block"
    assert head_grad is not None and head_grad.abs().sum() > 0, "no grad to head"
    print(f"  [ok] backward: loss={loss.item():.3f} parts={ {k: round(v,3) for k,v in parts.items()} }; "
          f"grads reach ego block + heads")


def test_ego_removes_global_shift():
    """Many static cells + one moving cell, all under the same global (ego)
    shift. Masked ego estimate should leave the moving cell's residual large
    and static residuals near zero."""
    torch.manual_seed(0)
    C = 8
    n_cells = 30
    cells = torch.stack([torch.zeros(n_cells, dtype=torch.long),
                         torch.arange(1, n_cells + 1),
                         torch.arange(1, n_cells + 1),
                         torch.arange(1, n_cells + 1)], dim=1)
    base = torch.randn(n_cells, C)
    global_shift = torch.randn(1, C)                 # ego-like, same everywhere
    local = torch.zeros(n_cells, C)
    local[0] = torch.randn(C) * 3                    # one truly-moving cell

    coords, feats = [], []
    for i in range(n_cells):
        coords.append(torch.cat([cells[i], torch.tensor([0])]))      # t=0
        feats.append(base[i] + global_shift[0] + local[i])
        coords.append(torch.cat([cells[i], torch.tensor([1])]))      # t=1
        feats.append(base[i])
    x = ST(torch.stack(feats), torch.stack(coords).long())

    app, valid = temporal_difference(x)
    bidx = x.coords[:, 0].long()
    B = int(bidx.max().item()) + 1
    ego = masked_mean_per_batch(app.feats, bidx, valid, B)[bidx]
    residual = app.feats - ego

    res_ref = residual[x.coords[:, 4] == 0]
    moving_mag = res_ref[0].abs().mean().item()
    static_mag = res_ref[1:].abs().mean().item()
    assert moving_mag > static_mag * 3, \
        f"ego removal failed: moving={moving_mag:.3f} static={static_mag:.3f}"
    print(f"  [ok] ego decoupling: residual on moving cell={moving_mag:.3f} "
          f">> static cells={static_mag:.3f}")


if __name__ == "__main__":
    print("Running model logic tests (mock backend)...")
    test_forward_backward()
    test_ego_removes_global_shift()
    print("All model checks passed.")
