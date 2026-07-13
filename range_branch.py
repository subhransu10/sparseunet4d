"""SalsaNext-lite 2D range-image motion branch.

Input : (B, C_in, 64, 2048)  -- range + K residual images + validity mask
Output: (B, C_out, 64, 2048) -- per-pixel motion-context features, gathered
        into the sparse backbone at head level via pixel_idx.

~2.5M params @ base=32. Height is only 64 px, so vertical stride is applied
once; horizontal stride twice. Decoder = bilinear upsample + conv.
Place at sparseunet4d/models/range_branch.py
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(i, o, k=3, s=1, d=1):
    p = d * (k // 2)
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = _cbr(ch, ch)
        self.c2 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(ch))
    def forward(self, x):
        return F.relu(x + self.c2(self.c1(x)), inplace=True)


class RangeBranch2D(nn.Module):
    """Enc: base -> 2b -> 4b with strides (1,2),(2,2),(1,2); dec back to full res."""
    def __init__(self, in_ch=5, base=32, out_ch=32):
        super().__init__()
        b = base
        self.stem = nn.Sequential(_cbr(in_ch, b), ResBlock2D(b))
        self.e1 = nn.Sequential(_cbr(b, 2*b, s=(1, 2)), ResBlock2D(2*b))
        self.e2 = nn.Sequential(_cbr(2*b, 4*b, s=(2, 2)), ResBlock2D(4*b))
        self.e3 = nn.Sequential(_cbr(4*b, 4*b, s=(1, 2), d=1), ResBlock2D(4*b),
                                _cbr(4*b, 4*b, d=2))          # dilated context
        self.d2 = _cbr(4*b + 4*b, 2*b)
        self.d1 = _cbr(2*b + 2*b, b)
        self.d0 = _cbr(b + b, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x):
        s0 = self.stem(x)          # (b,  64, 2048)
        s1 = self.e1(s0)           # (2b, 64, 1024)
        s2 = self.e2(s1)           # (4b, 32,  512)
        s3 = self.e3(s2)           # (4b, 32,  256)
        h = self.d2(torch.cat([self._up(s3, s2), s2], 1))
        h = self.d1(torch.cat([self._up(h, s1), s1], 1))
        h = self.d0(torch.cat([self._up(h, s0), s0], 1))
        return self.out(h)         # (out_ch, 64, 2048)


def gather_pixel_features(f2d, pixel_idx, batch_index, out_ch):
    """f2d: (B,C,H,W); pixel_idx: (N,) flat v*W+u or -1; batch_index: (N,) long.
    Returns (N, C) with zeros where pixel_idx < 0. Backend-agnostic."""
    B, C, H, W = f2d.shape
    g = f2d.new_zeros(pixel_idx.shape[0], out_ch)
    valid = pixel_idx >= 0
    if valid.any():
        flat = f2d.permute(0, 2, 3, 1).reshape(B, H * W, C)
        g[valid] = flat[batch_index[valid], pixel_idx[valid]]
    return g