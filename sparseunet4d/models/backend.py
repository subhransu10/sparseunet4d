"""Sparse-op abstraction with two interchangeable backends.

- backend='mock'  : pure-CPU torch. NOT a physically faithful sparse conv;
                    it preserves tensor shapes, coordinate bookkeeping, and
                    gradient flow so all model/loss/eval LOGIC is testable on a
                    laptop GPU (or CPU) without MinkowskiEngine.
- backend='me'    : MinkowskiEngine (4D). Same network code; swap on the 5090.

Coordinate layout everywhere: coords[:, 0]=batch, [1:4]=x,y,z (spatial),
[4]=t (temporal). We never stride t (spatial-only downsampling lives here).

Set backend via env SU4D_BACKEND=me|mock or set_backend(...).
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn

_BACKEND = os.environ.get("SU4D_BACKEND", "mock")


def set_backend(name: str):
    global _BACKEND
    assert name in ("mock", "me")
    _BACKEND = name


def backend() -> str:
    return _BACKEND


# ----------------------------------------------------------------------------
# tensor wrapper (mock backend). For 'me', we wrap ME.SparseTensor instead.
# ----------------------------------------------------------------------------
class ST:
    """Minimal sparse-tensor wrapper for the mock backend."""
    __slots__ = ("feats", "coords")

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor):
        self.feats = feats           # (N, C) float
        self.coords = coords         # (N, 5) long [b, x, y, z, t]

    def like(self, feats):
        return ST(feats, self.coords)


def _scatter_mean(src, index, num):
    C = src.shape[1]
    out = torch.zeros(num, C, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    cnt = torch.zeros(num, dtype=src.dtype, device=src.device)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1.0).unsqueeze(1)


# ----------------------------------------------------------------------------
# Modules. Each implements a mock path and (structurally) an ME path.
# ----------------------------------------------------------------------------
class SparseConv(nn.Module):
    """4D sparse conv. stride applies to spatial dims only (t stride fixed=1)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dim=4):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        if _BACKEND == "me":
            import MinkowskiEngine as ME
            self.conv = ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=kernel_size,
                stride=(stride, stride, stride, 1), dimension=dim)
        else:
            self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        if _BACKEND == "me":
            return self.conv(x)
        # mock
        f = self.lin(x.feats)
        if self.stride == 1:
            return ST(f, x.coords)
        # spatial-only downsample: divide x,y,z by stride; keep b, t
        c = x.coords.clone()
        c[:, 1:4] = torch.div(c[:, 1:4], self.stride, rounding_mode="floor")
        keys, inv = torch.unique(c, dim=0, return_inverse=True)
        f_ds = _scatter_mean(f, inv, keys.shape[0])
        return ST(f_ds, keys)


class SparseConvTranspose(nn.Module):
    """Upsample features onto a finer template tensor's coordinates."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, dim=4):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        if _BACKEND == "me":
            import MinkowskiEngine as ME
            self.conv = ME.MinkowskiConvolutionTranspose(
                in_ch, out_ch, kernel_size=kernel_size,
                stride=(stride, stride, stride, 1), dimension=dim)
        else:
            self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, x, template):
        if _BACKEND == "me":
            return self.conv(x, coordinates=template.coordinate_map_key)
        # mock: map each fine voxel to its coarse parent and gather feats
        f = self.lin(x.feats)                       # (Ncoarse, out)
        parent = template.coords.clone()
        parent[:, 1:4] = torch.div(parent[:, 1:4], self.stride,
                                   rounding_mode="floor")
        # build lookup from coarse coord -> row index
        keys = x.coords
        cat = torch.cat([keys, parent], dim=0)
        uniq, inv = torch.unique(cat, dim=0, return_inverse=True)
        coarse_row = torch.full((uniq.shape[0],), -1, dtype=torch.long)
        coarse_row[inv[:keys.shape[0]]] = torch.arange(keys.shape[0])
        fine_to_coarse = coarse_row[inv[keys.shape[0]:]]
        valid = fine_to_coarse >= 0
        out = torch.zeros(template.coords.shape[0], self.out_ch, dtype=f.dtype)
        out[valid] = f[fine_to_coarse[valid]]
        return ST(out, template.coords)


class SparseBN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        if _BACKEND == "me":
            import MinkowskiEngine as ME
            self.bn = ME.MinkowskiBatchNorm(ch)
        else:
            self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        if _BACKEND == "me":
            return self.bn(x)
        return x.like(self.bn(x.feats))


class SparseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        if _BACKEND == "me":
            import MinkowskiEngine as ME
            self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        if _BACKEND == "me":
            return self.relu(x)
        return x.like(self.relu(x.feats))


def global_avg_pool_per_batch(x):
    """Return (B, C) batch-wise mean of features, and the batch index per row."""
    if _BACKEND == "me":
        feats, coords = x.F, x.C
    else:
        feats, coords = x.feats, x.coords
    b = coords[:, 0].long()
    B = int(b.max().item()) + 1
    pooled = _scatter_mean(feats, b, B)
    return pooled, b


def broadcast_scale(x, scale, batch_index):
    """Multiply each voxel's features by its batch's scale vector (SE excite)."""
    s = scale[batch_index]
    if _BACKEND == "me":
        return type(x)(x.F * s, coordinate_map_key=x.coordinate_map_key,
                       coordinate_manager=x.coordinate_manager)
    return x.like(x.feats * s)


def cat_features(a, b):
    """Channel-concat two tensors that share coordinates (skip connection)."""
    if _BACKEND == "me":
        import MinkowskiEngine as ME
        return ME.cat(a, b)
    assert torch.equal(a.coords, b.coords), "skip coords misaligned"
    return a.like(torch.cat([a.feats, b.feats], dim=1))


def masked_mean_per_batch(feats, batch_index, mask, B):
    """Mean of feats over rows where mask is True, grouped by batch."""
    m = mask.float().unsqueeze(1)
    num = torch.zeros(B, feats.shape[1], dtype=feats.dtype, device=feats.device)
    num.index_add_(0, batch_index, feats * m)
    den = torch.zeros(B, 1, dtype=feats.dtype, device=feats.device)
    den.index_add_(0, batch_index, m)
    return num / den.clamp(min=1.0)


def temporal_difference(x):
    """Per spatial-cell apparent motion: feat(t=0) - mean(feat(t>0)).

    Returns (diff_tensor_on_x_coords, valid_mask). `valid_mask` marks reference
    voxels (t==0) that have at least one temporal partner -- only these carry a
    meaningful apparent-motion signal and should feed the ego estimate.
    """
    if _BACKEND == "me":
        feats, coords = x.F, x.C
        coords = coords.to(feats.device) 
    else:
        feats, coords = x.feats, x.coords
    spatial = coords[:, :4]                      # [b, x, y, z]
    t = coords[:, 4]
    keys, inv = torch.unique(spatial, dim=0, return_inverse=True)
    G = keys.shape[0]
    is_ref = (t == 0)
    is_ctx = ~is_ref
    ctx_mean = _scatter_mean(feats[is_ctx], inv[is_ctx], G) \
        if is_ctx.any() else torch.zeros(G, feats.shape[1], device=feats.device)
    has_ctx = torch.zeros(G, dtype=torch.bool, device=feats.device)
    has_ctx[inv[is_ctx]] = True
    diff = torch.zeros_like(feats)
    valid = torch.zeros(feats.shape[0], dtype=torch.bool, device=feats.device)
    ref_rows = torch.nonzero(is_ref, as_tuple=False).squeeze(1)
    ref_cells = inv[ref_rows]
    ok = has_ctx[ref_cells]
    rows_ok = ref_rows[ok]
    diff[rows_ok] = feats[rows_ok] - ctx_mean[ref_cells[ok]]
    valid[rows_ok] = True
    if _BACKEND == "me":
        diff_st = type(x)(diff, coordinate_map_key=x.coordinate_map_key,
                          coordinate_manager=x.coordinate_manager)
    else:
        diff_st = x.like(diff)
    return diff_st, valid
