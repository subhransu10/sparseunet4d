"""DualBranchUNet4D (P2): decoupled motion / appearance encoders.

Why: today the residual (motion) channels are concatenated onto the occupancy/
remission channel and a SINGLE backbone must disentangle "this looks like a car"
from "this moved". Evidence it is overloaded: semantic mIoU plateaus ~0.42 while
motion IoU climbs. MF-MOS (76.1) and CV-MOS (77.5) both run a DEDICATED motion
encoder alongside the appearance encoder and fuse late.

Design (mirrors that, adapted to 4D sparse voxels):
  appearance branch : remission/occupancy channels  -> stem_a -> encoders_a
  motion branch     : K residual channels           -> stem_m -> encoders_m
  fuse at bottleneck: cat(a_deep, m_deep) -> bottleneck
  decoder           : up(h) + cat(skip_a[i], skip_m[i])  (both branches' skips)

Heads (motion / semantic / offset) and the cluster-consistency head are
unchanged, so this is a clean one-variable swap against cluster_v3.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .backend import (SparseConv, SparseConvTranspose, cat_features, backend)
from .model import SEResBlock4D
from .cluster_head import ClusterConsistencyHead


def _split_channels(x, n_first):
    """Split a sparse tensor's features into (first n_first ch, remaining ch),
    keeping the SAME coordinates/coordinate manager."""
    if backend() == "me":
        import MinkowskiEngine as ME
        a = ME.SparseTensor(x.F[:, :n_first].contiguous(),
                            coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=x.coordinate_manager)
        b = ME.SparseTensor(x.F[:, n_first:].contiguous(),
                            coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=x.coordinate_manager)
        return a, b
    from .backend import ST
    return (ST(x.feats[:, :n_first], x.coords),
            ST(x.feats[:, n_first:], x.coords))


class DualBranchUNet4D(nn.Module):
    def __init__(self, in_ch=1, num_semantic=20, base=32, n_stages=2, use_se=True,
                 app_ch=1, use_cluster=False, cluster_link=2, cluster_min_size=3,
                 cluster_cross_frame=False, cluster_feature_fusion=False):
        super().__init__()
        self.n_stages = n_stages
        self.app_ch = app_ch                      # appearance channels (remission)
        mot_ch = in_ch - app_ch                   # residual/motion channels
        assert mot_ch > 0, "dual branch needs residual channels (residual_feats: true)"
        w = [base * (2 ** i) for i in range(n_stages + 1)]

        self.stem_a = SEResBlock4D(app_ch, w[0], 1, use_se)
        self.stem_m = SEResBlock4D(mot_ch, w[0], 1, use_se)
        self.encoders_a = nn.ModuleList(
            [SEResBlock4D(w[i], w[i + 1], 2, use_se) for i in range(n_stages)])
        self.encoders_m = nn.ModuleList(
            [SEResBlock4D(w[i], w[i + 1], 2, use_se) for i in range(n_stages)])
        # fuse the two branches at the bottleneck
        self.bottleneck = SEResBlock4D(2 * w[-1], w[-1], 1, use_se)
        self.ups = nn.ModuleList(
            [SparseConvTranspose(w[i + 1], w[i], 3, 2) for i in range(n_stages)])
        # decoder input: up(h)=w[i]  +  skip_a=w[i] + skip_m=w[i]
        self.decoders = nn.ModuleList(
            [SEResBlock4D(3 * w[i], w[i], 1, use_se) for i in range(n_stages)])

        self.motion_head = nn.Linear(w[0], 2)
        self.semantic_head = nn.Linear(w[0], num_semantic)
        self.offset_head = nn.Sequential(
            nn.Linear(w[0], w[0]), nn.ReLU(inplace=True), nn.Linear(w[0], 3))
        self.cluster = (ClusterConsistencyHead(w[0], cluster_link, cluster_min_size,
                                               cluster_cross_frame,
                                               cluster_feature_fusion)
                        if use_cluster else None)
        self.register_buffer("_movable", torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                             persistent=False)

    def forward(self, x):
        xa, xm = _split_channels(x, self.app_ch)
        sa = [self.stem_a(xa)]
        sm = [self.stem_m(xm)]
        for i in range(self.n_stages):
            sa.append(self.encoders_a[i](sa[-1]))
            sm.append(self.encoders_m[i](sm[-1]))

        h = self.bottleneck(cat_features(sa[-1], sm[-1]))
        for i in reversed(range(self.n_stages)):
            h = self.ups[i](h, sa[i])
            h = self.decoders[i](cat_features(h, cat_features(sa[i], sm[i])))

        feats = h.feats if backend() != "me" else h.F
        coords = h.coords if backend() != "me" else h.C
        motion_logits = self.motion_head(feats)
        semantic_logits = self.semantic_head(feats)
        out = {"motion_logits": motion_logits,
               "semantic_logits": semantic_logits,
               "offset_pred": self.offset_head(feats),
               "coords": coords}
        if self.cluster is not None:
            sem_arg = semantic_logits.argmax(1)
            fg = torch.isin(sem_arg, self._movable.to(sem_arg.device))
            fused, clog, rid = self.cluster(feats, coords, motion_logits, fg)
            out["motion_logits"] = fused
            out["cluster_logits"] = clog
            out["cluster_row_id"] = rid
        return out
