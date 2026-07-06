"""SparseUNet4D — configurable width (base) and depth (n_stages).

Spatial-only downsampling U-Net for joint MOS + semantic. Ego-decouple removed
by default (ablation proved it adds nothing). Scale via config:
  model: { base: 64, n_stages: 3, use_se: true }
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .backend import (SparseConv, SparseConvTranspose, SparseBN, SparseReLU,
                      global_avg_pool_per_batch, broadcast_scale, cat_features,
                      backend)


class SEModule(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, h); self.fc2 = nn.Linear(h, ch)

    def forward(self, x):
        z, bi = global_avg_pool_per_batch(x)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))
        return broadcast_scale(x, s, bi)


class SEResBlock4D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = SparseConv(in_ch, out_ch, 3, stride); self.bn1 = SparseBN(out_ch)
        self.conv2 = SparseConv(out_ch, out_ch, 3, 1); self.bn2 = SparseBN(out_ch)
        self.relu = SparseReLU()
        self.se = SEModule(out_ch) if use_se else None
        self.proj = SparseConv(in_ch, out_ch, 1, stride) if (in_ch != out_ch or stride != 1) else None

    def forward(self, x):
        idn = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None: out = self.se(out)
        out = out.like(out.feats + idn.feats) if backend() != "me" else out + idn
        return self.relu(out)


class SparseUNet4D(nn.Module):
    def __init__(self, in_ch=1, num_semantic=20, base=32, n_stages=2, use_se=True,
                 use_ego_decouple=False):
        super().__init__()
        self.n_stages = n_stages
        widths = [base * (2 ** i) for i in range(n_stages + 1)]   # w0..w_n
        self.stem = SEResBlock4D(in_ch, widths[0], 1, use_se)
        self.encoders = nn.ModuleList(
            [SEResBlock4D(widths[i], widths[i + 1], 2, use_se) for i in range(n_stages)])
        self.bottleneck = SEResBlock4D(widths[-1], widths[-1], 1, use_se)
        self.ups = nn.ModuleList(
            [SparseConvTranspose(widths[i + 1], widths[i], 3, 2) for i in range(n_stages)])
        self.decoders = nn.ModuleList(
            [SEResBlock4D(2 * widths[i], widths[i], 1, use_se) for i in range(n_stages)])
        self.motion_head = nn.Linear(widths[0], 2)
        self.semantic_head = nn.Linear(widths[0], num_semantic)

    def forward(self, x):
        s = [self.stem(x)]
        for enc in self.encoders:
            s.append(enc(s[-1]))
        h = self.bottleneck(s[-1])
        for i in reversed(range(self.n_stages)):
            h = self.ups[i](h, s[i])
            h = self.decoders[i](cat_features(h, s[i]))
        feats = h.feats if backend() != "me" else h.F
        return {"motion_logits": self.motion_head(feats),
                "semantic_logits": self.semantic_head(feats),
                "coords": h.coords if backend() != "me" else h.C}