"""Object-level moving-consistency head (P1).

Thesis link: training pastes movers WITH their multi-frame trajectories
(trajectory-consistent injection); this head enforces the SAME rigid-object
prior at inference -- a car/person moves as one unit, so all its points share
one moving label. It clusters foreground reference-frame (t=0) voxels, pools the
backbone features per object, predicts ONE moving logit per cluster, and adds it
(through a learnable gate initialised to 0) to the per-voxel moving logit. This
recovers partial/occluded object parts the per-voxel head misses -- directly
attacking the recall bottleneck. Mirrors 4D-CS / InsMOS cluster priors.

Drop-in: sparseunet4d/models/cluster_head.py . Pure numpy/scipy + torch.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def connected_components_vox(coords_xyz: np.ndarray, link: int = 2,
                             min_size: int = 3) -> np.ndarray:
    """Cluster integer voxel coords by Chebyshev-<=`link` connectivity.

    coords_xyz : (M,3) int voxel indices for ONE batch item, foreground only.
    Returns    : (M,) int64 cluster id; -1 for voxels whose cluster is smaller
                 than `min_size` (noise). CPU, pure numpy/scipy.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    M = len(coords_xyz)
    if M == 0:
        return np.zeros(0, np.int64)
    c = coords_xyz.astype(np.int64)
    # hash each occupied voxel; sort keys once for vectorized neighbour lookup
    key = (c[:, 0] + 2**20) * 2**42 + (c[:, 1] + 2**20) * 2**21 + (c[:, 2] + 2**20)
    order = np.argsort(key)
    key_s = key[order]
    rows, cols = [], []
    for dx in range(-link, link + 1):
        for dy in range(-link, link + 1):
            for dz in range(-link, link + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nkey = ((c[:, 0] + dx + 2**20) * 2**42
                        + (c[:, 1] + dy + 2**20) * 2**21
                        + (c[:, 2] + dz + 2**20))
                pos = np.clip(np.searchsorted(key_s, nkey), 0, M - 1)
                hit = key_s[pos] == nkey                 # neighbour exists?
                rows.append(np.nonzero(hit)[0])
                cols.append(order[pos[hit]])
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if len(cols) else np.zeros(0, np.int64)
    if len(rows):
        g = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(M, M))
        _, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(M, dtype=np.int64)
    labels = labels.astype(np.int64)
    # drop clusters smaller than min_size (noise)
    uniq, cnt = np.unique(labels, return_counts=True)
    small = uniq[cnt < min_size]
    if len(small):
        labels[np.isin(labels, small)] = -1
    return labels


class ClusterConsistencyHead(nn.Module):
    """Pool backbone features per foreground object, predict a per-cluster moving
    logit, fuse it into the per-voxel moving logit through a gate (init 0)."""

    def __init__(self, feat_dim: int, link: int = 2, min_size: int = 3,
                 cross_frame: bool = False, feature_fusion: bool = False):
        super().__init__()
        # per-cluster moving logit (aux BCE loss shapes the pooled features)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1))
        self.feature_fusion = feature_fusion
        if feature_fusion:
            # concat(own feat, object feat) -> motion-logit delta (4D-CS-style)
            self.fusion = nn.Sequential(
                nn.Linear(2 * feat_dim, feat_dim), nn.ReLU(inplace=True),
                nn.Linear(feat_dim, 2))
        self.gate = nn.Parameter(torch.zeros(1))   # 0 -> no effect until learned
        self.link, self.min_size = link, min_size
        self.cross_frame = cross_frame

    def forward(self, feats: torch.Tensor, coords5: torch.Tensor,
                motion_logits: torch.Tensor, fg_mask: torch.Tensor):
        """
        feats         : (N, C) finest-level voxel features
        coords5       : (N, 5) int [b, x, y, z, t]
        motion_logits : (N, 2) per-voxel static/moving logits
        fg_mask       : (N,) bool  foreground (movable-class) voxels
        Returns fused motion_logits (N,2), cluster_logits (G,) or None,
                loss_row_id (N,) long: cluster id for REFERENCE (t=0) members,
                -1 elsewhere (used for the per-cluster loss).

        cross_frame=False (v1): cluster t=0 foreground only.
        cross_frame=True  (v2): cluster foreground over ALL frames (ego-
        registered), pool features across every member (enriching sparse/occluded
        t=0 objects with their past-frame observations), but bias + supervise only
        the reference (t=0) members that we actually score.
        """
        device = feats.device
        N = feats.shape[0]
        cn = coords5.detach().cpu().numpy()
        t_is0 = (coords5[:, 4] == 0)
        cluster_set = fg_mask if self.cross_frame else (t_is0 & fg_mask)
        csn = cluster_set.cpu().numpy()

        # cluster per batch item -> a global cluster id per member voxel (row_cid)
        row_cid = torch.full((N,), -1, dtype=torch.long, device=device)
        next_id = 0
        for bb in (np.unique(cn[csn, 0]) if csn.any() else []):
            sel = np.nonzero(csn & (cn[:, 0] == bb))[0]
            labels = connected_components_vox(cn[sel, 1:4], self.link, self.min_size)
            valid = labels >= 0
            if valid.any():
                gl = labels.copy()
                gl[valid] += next_id
                next_id += int(labels[valid].max()) + 1
                rid = np.full(len(sel), -1, np.int64)
                rid[valid] = gl[valid]
                row_cid[torch.from_numpy(sel).to(device)] = \
                    torch.from_numpy(rid).to(device)

        G = next_id
        loss_row_id = torch.full((N,), -1, dtype=torch.long, device=device)
        if G == 0:
            return motion_logits, None, loss_row_id

        # pool features over ALL members of each cluster (all frames if cross_frame)
        member = row_cid >= 0
        ids = row_cid[member]
        C = feats.shape[1]
        csum = torch.zeros(G, C, device=device).index_add_(0, ids, feats[member])
        ccnt = torch.zeros(G, device=device).index_add_(
            0, ids, torch.ones_like(ids, dtype=feats.dtype))
        cfeat = csum / ccnt.clamp(min=1.0).unsqueeze(1)
        clog = self.mlp(cfeat).squeeze(1)                       # (G,)

        # fuse + supervise only on reference (t=0) members
        ref = t_is0 & member
        if self.feature_fusion:
            cf_ref = cfeat[row_cid[ref]]                     # (Nref, C) object feat
            delta = self.fusion(torch.cat([feats[ref], cf_ref], dim=1))   # (Nref,2)
            delta_full = torch.zeros_like(motion_logits)
            delta_full[ref] = delta
            fused = motion_logits + self.gate * delta_full   # gate=0 -> baseline
        else:
            bias = torch.zeros(N, device=device)
            bias[ref] = self.gate * clog[row_cid[ref]]       # v1 scalar bias
            fused = torch.stack([motion_logits[:, 0],
                                 motion_logits[:, 1] + bias], dim=1)
        loss_row_id[ref] = row_cid[ref]
        return fused, clog, loss_row_id
