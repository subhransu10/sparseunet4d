"""T-ladder step 2: paint a pixel, verify the right voxel receives it.

Catches: u/v swap, flat-index math (v*W+u), batch-index misrouting,
past-frame (-1) leakage. CPU, no data needed.

  PYTHONPATH=~/sparseunet4d python3 test_gather_roundtrip.py
"""
import torch, numpy as np, os, sys
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.models.range_branch import gather_pixel_features

H, W, C = 64, 2048, 32

def test_basic_routing():
    B = 2
    f2d = torch.zeros(B, C, H, W)
    # paint two known pixels with distinct signatures
    f2d[0, :, 10, 500] = 1.0
    f2d[1, :, 33, 1999] = 2.0
    # voxels: [batch0 painted, batch0 other pixel, batch0 past(-1), batch1 painted]
    pixel_idx = torch.tensor([10 * W + 500, 5 * W + 7, -1, 33 * W + 1999])
    b = torch.tensor([0, 0, 0, 1])
    g = gather_pixel_features(f2d, pixel_idx, b, C)
    assert torch.allclose(g[0], torch.full((C,), 1.0)), "batch0 gather wrong"
    assert torch.allclose(g[1], torch.zeros(C)), "unpainted pixel not zero"
    assert torch.allclose(g[2], torch.zeros(C)), "-1 (past frame) leaked"
    assert torch.allclose(g[3], torch.full((C,), 2.0)), "batch routing wrong"
    print("PASS routing")

def test_uv_convention_matches_dataset():
    """flat = v*W + u must invert to (v, u)."""
    v, u = 42, 1337
    flat = v * W + u
    f2d = torch.zeros(1, C, H, W); f2d[0, :, v, u] = 7.0
    g = gather_pixel_features(f2d, torch.tensor([flat]), torch.tensor([0]), C)
    assert torch.allclose(g[0], torch.full((C,), 7.0)), "v*W+u convention broken"
    print("PASS uv convention")

def test_gradient_flows():
    f2d = torch.zeros(1, C, H, W, requires_grad=True)
    g = gather_pixel_features(f2d, torch.tensor([3 * W + 3, -1]),
                              torch.tensor([0, 0]), C)
    g.sum().backward()
    assert f2d.grad[0, :, 3, 3].abs().sum() > 0, "no grad at gathered pixel"
    assert f2d.grad.abs().sum() == f2d.grad[0, :, 3, 3].abs().sum(), \
        "grad leaked to non-gathered pixels"
    print("PASS gradient")

def test_branch_shapes():
    from sparseunet4d.models.range_branch import RangeBranch2D
    m = RangeBranch2D(in_ch=5, base=32, out_ch=32)
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 5, H, W))
    assert y.shape == (2, 32, H, W), y.shape
    print(f"PASS branch shapes ({n/1e6:.2f}M params)")

if __name__ == "__main__":
    test_basic_routing()
    test_uv_convention_matches_dataset()
    test_gradient_flows()
    test_branch_shapes()
    print("\nALL PASS")