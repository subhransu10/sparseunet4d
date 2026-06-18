"""SparseUNet4D model.

Components:
  - SEModule           : channel attention via batch-wise global pooling.
  - SEResBlock4D       : residual 4D sparse block + SE (the 'old paper' block).
  - EgoDecoupleBlock   : THE CONTRIBUTION. Splits apparent motion into a globally
                         consistent (ego-induced) component and a local residual
                         (object motion), and injects only the residual. Under
                         odometry drift the ego component is large but is removed,
                         so the motion cue stops collapsing into false positives.
  - SparseUNet4D       : spatial-only-downsampling U-Net, dual motion+semantic heads.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .backend import (
    SparseConv, SparseConvTranspose, SparseBN, SparseReLU,
    global_avg_pool_per_batch, broadcast_scale, cat_features,
    temporal_difference, masked_mean_per_batch, backend,
)


class SEModule(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(ch // r, 4))
        self.fc2 = nn.Linear(max(ch // r, 4), ch)

    def forward(self, x):
        z, bidx = global_avg_pool_per_batch(x)      # (B, C)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))
        return broadcast_scale(x, s, bidx)


class SEResBlock4D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = SparseConv(in_ch, out_ch, 3, stride)
        self.bn1 = SparseBN(out_ch)
        self.conv2 = SparseConv(out_ch, out_ch, 3, 1)
        self.bn2 = SparseBN(out_ch)
        self.relu = SparseReLU()
        self.se = SEModule(out_ch) if use_se else None
        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = SparseConv(in_ch, out_ch, 1, stride)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out = out.like(out.feats + identity.feats) if backend() != "me" \
            else out + identity
        return self.relu(out)


class EgoDecoupleBlock(nn.Module):
    """Separate global (ego) from local (object) apparent motion.

    apparent      = feat(t0) - mean(feat(t>0))           [temporal difference]
    ego_component = MLP(global_pool(apparent))           [globally consistent]
    residual      = apparent - broadcast(ego_component)  [object motion]
    gate          = sigmoid(MLP([feat, residual]))
    out           = feat + gate * Linear(residual)

    Rationale: imperfect ego-motion produces an apparent shift that is roughly
    consistent across all static voxels; true object motion is local. Removing
    the global component before forming the motion cue makes the cue robust to
    registration error.
    """
    def __init__(self, ch):
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.gate = nn.Sequential(nn.Linear(2 * ch, ch), nn.ReLU(),
                                  nn.Linear(ch, ch), nn.Sigmoid())
        self.proj = nn.Linear(ch, ch)

    def forward(self, x):
        app, valid = temporal_difference(x)               # ST, mask
        app_f = app.feats if backend() != "me" else app.F
        coords = x.coords if backend() != "me" else x.C
        bidx = coords[:, 0].long()
        B = int(bidx.max().item()) + 1
        ego_est = masked_mean_per_batch(app_f, bidx, valid, B)  # (B, C)
        ego = self.ego(ego_est)[bidx]                     # broadcast to voxels
        residual = app_f - ego                            # object motion
        feat = x.feats if backend() != "me" else x.F
        gate = self.gate(torch.cat([feat, residual], dim=1))
        out_f = feat + gate * self.proj(residual)
        return x.like(out_f) if backend() != "me" else \
            type(x)(out_f, coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager)


class SparseUNet4D(nn.Module):
    def __init__(self, in_ch=1, num_semantic=20, base=32,
                 use_se=True, use_ego_decouple=True):
        super().__init__()
        self.stem = SEResBlock4D(in_ch, base, 1, use_se)
        self.enc1 = SEResBlock4D(base, base * 2, 2, use_se)
        self.enc2 = SEResBlock4D(base * 2, base * 4, 2, use_se)
        self.bottleneck = SEResBlock4D(base * 4, base * 4, 1, use_se)
        self.ego = EgoDecoupleBlock(base * 4) if use_ego_decouple else None
        self.up2 = SparseConvTranspose(base * 4, base * 2, 3, 2)
        self.dec2 = SEResBlock4D(base * 4, base * 2, 1, use_se)   # +skip
        self.up1 = SparseConvTranspose(base * 2, base, 3, 2)
        self.dec1 = SEResBlock4D(base * 2, base, 1, use_se)       # +skip
        self.motion_head = nn.Linear(base, 2)
        self.semantic_head = nn.Linear(base, num_semantic)

    def forward(self, x):
        s0 = self.stem(x)            # full res
        s1 = self.enc1(s0)           # /2
        s2 = self.enc2(s1)           # /4
        b = self.bottleneck(s2)
        if self.ego is not None:
            b = self.ego(b)
        d2 = self.up2(b, s1)         # back to /2 coords
        d2 = self.dec2(cat_features(d2, s1))
        d1 = self.up1(d2, s0)        # back to full res
        d1 = self.dec1(cat_features(d1, s0))
        feats = d1.feats if backend() != "me" else d1.F
        return {
            "motion_logits": self.motion_head(feats),
            "semantic_logits": self.semantic_head(feats),
            "coords": d1.coords if backend() != "me" else d1.C,
        }
