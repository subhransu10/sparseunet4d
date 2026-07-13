"""SemanticKITTI 4D dataset: pose-aware multi-frame stacking + voxelization.

Returns a 4D sparse sample (x, y, z, t) for MinkowskiEngine. Only the reference
frame (t == 0) is supervised; past frames carry IGNORE_INDEX and exist purely
for temporal context.

Crucially, registration of past frames uses a *pluggable* PoseProvider. Swap
`pose_mode='gt'` -> 'drift' (with rot/trans std) to run the robustness study
WITHOUT retraining: same checkpoint, corrupted poses at inference.
"""

from __future__ import annotations
import os
from scipy import ndimage
import numpy as np
from torch.utils.data import Dataset
from .residual_features import residual_channels
from .poses import build_pose_provider
from .label_map import (
    load_semantic_learning_map, split_label,
    to_motion_labels, to_semantic_labels, IGNORE_INDEX,
)


def _read_scan(path: str) -> np.ndarray:
    """velodyne .bin -> (N, 4) [x, y, z, remission]."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def _read_label(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32)


def _transform(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 SE(3) to (N,3) points."""
    hom = np.concatenate([points_xyz, np.ones((len(points_xyz), 1))], axis=1)
    return (hom @ T.T)[:, :3]


# ============================================================================
# PATCH for sparseunet4d/datasets/semantickitti.py  (panoptic GT offsets)
# ============================================================================
# 1) REPLACE the whole _gt_offsets function with _gt_offsets_panoptic below.
# 2) In __getitem__, the t_idx==0 branch changes: split_label now keeps inst,
#    and the offset call uses (sem_raw, inst) instead of connected components.
# ============================================================================

# ---- (1) module-level: replace _gt_offsets with this ----------------------
def _gt_offsets_panoptic(xyz_m, sem_raw, inst_raw):
    """Per-point offset (meters) to its REAL instance center, for ALL thing
    points (parked + moving). Mask = thing points with a valid instance.

    Grouping key is the (semantic, instance) PAIR: instance ids can repeat
    across classes. Stuff points (inst==0) are unmasked with zero offset.
    """
    THING_RAW = {10, 11, 13, 15, 18, 20, 30, 31, 32,
                 252, 253, 254, 255, 256, 257, 258, 259}
    off = np.zeros((len(xyz_m), 3), dtype=np.float32)
    omask = np.zeros(len(xyz_m), dtype=bool)
    thing = np.isin(sem_raw, list(THING_RAW)) & (inst_raw > 0)
    idx = np.where(thing)[0]
    if len(idx) == 0:
        return off, omask
    key = sem_raw[idx].astype(np.int64) * 100000 + inst_raw[idx].astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    bounds = np.r_[starts, len(ks)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = idx[order[a:b]]
        off[sel] = xyz_m[sel].mean(0) - xyz_m[sel]
    omask[idx] = True
    return off, omask


class SemanticKITTI4D(Dataset):
    def __init__(self, root, sequences, n_frames=4, voxel_size=0.1,
                 semantic_yaml=None, pose_mode="gt",
                 rot_std_deg=0.0, trans_std_m=0.0, pose_seed=0,
                 point_range=51.2, max_points=None,
                 residual_feats=True, res_clip=3.0):
        """
        root:        .../sequences
        sequences:   list of int (e.g. list(range(11)) for train/val)
        n_frames:    number of stacked frames including the reference
        voxel_size:  spatial quantization in metres
        pose_mode:   'gt' or 'drift'
        point_range: clip |x|,|y|,|z| beyond this (m); None to disable
        """
        self.root = root
        self.n_frames = n_frames
        self.voxel_size = voxel_size
        self.pose_mode = pose_mode
        self.rot_std_deg = rot_std_deg
        self.trans_std_m = trans_std_m
        self.pose_seed = pose_seed
        self.point_range = point_range
        self.max_points = max_points
        self.residual_feats = residual_feats
        self.res_clip = res_clip

        self.lut = None
        if semantic_yaml is not None:
            self.lut, self.lmap_inv, self.num_sem = \
                load_semantic_learning_map(semantic_yaml)

        # index = list of (seq, ref_frame_idx); cache one pose provider per seq
        self.index = []
        self.pose_providers = {}
        for seq in sequences:
            seq_dir = os.path.join(root, f"{seq:02d}")
            velo_dir = os.path.join(seq_dir, "velodyne")
            n = len([f for f in os.listdir(velo_dir) if f.endswith(".bin")])
            self.pose_providers[seq] = build_pose_provider(
                seq_dir, pose_mode, rot_std_deg, trans_std_m, pose_seed)
            for ref in range(n):
                self.index.append((seq, ref))

    def __len__(self):
        return len(self.index)

    def _frame_paths(self, seq, frame):
        seq_dir = os.path.join(self.root, f"{seq:02d}")
        bin_p = os.path.join(seq_dir, "velodyne", f"{frame:06d}.bin")
        lab_p = os.path.join(seq_dir, "labels", f"{frame:06d}.label")
        return bin_p, lab_p

    def __getitem__(self, i):
        seq, ref = self.index[i]
        provider = self.pose_providers[seq]
        # frames: ref, ref-1, ..., ref-(n_frames-1) ; t index 0 == reference
        frames = [ref - k for k in range(self.n_frames)]
        frames = [f for f in frames if f >= 0]

        coords_all, feats_all, mot_all, sem_all = [], [], [], []
        off_all, omask_all = [], []
        frame_xyz = []  # keep clipped xyz per frame for residual computation
        for t_idx, f in enumerate(frames):
            bin_p, lab_p = self._frame_paths(seq, f)
            scan = _read_scan(bin_p)
            xyz, remission = scan[:, :3], scan[:, 3:4]

            # register past frame into reference frame using (possibly drifted) pose
            if t_idx != 0:
                T = provider.relative(f, ref)
                xyz = _transform(xyz, T).astype(np.float32)

            # range clip
            if self.point_range is not None:
                m = np.all(np.abs(xyz) < self.point_range, axis=1)
                xyz, remission = xyz[m], remission[m]
            else:
                m = np.ones(len(xyz), dtype=bool)

            # labels: supervise only the reference frame
            if t_idx == 0 and os.path.exists(lab_p):
                sem_raw, inst_raw = split_label(_read_label(lab_p))   # keep inst now
                sem_raw = sem_raw[m]
                inst_raw = inst_raw[m]                                # clip same mask
                mot = to_motion_labels(sem_raw)
                sem = (to_semantic_labels(sem_raw, self.lut)
                       if self.lut is not None
                       else np.full(len(sem_raw), IGNORE_INDEX))
                off, omask = _gt_offsets_panoptic(xyz, sem_raw, inst_raw)
            else:
                mot = np.full(len(xyz), IGNORE_INDEX, dtype=np.int64)
                sem = np.full(len(xyz), IGNORE_INDEX, dtype=np.int64)
                off = np.zeros((len(xyz), 3), np.float32)
                omask = np.zeros(len(xyz), bool)

            frame_xyz.append(xyz)
            t_col = np.full((len(xyz), 1), t_idx, dtype=np.float32)
            coords_all.append(np.concatenate([xyz, t_col], axis=1))
            feats_all.append(remission)
            mot_all.append(mot)
            sem_all.append(sem)
            off_all.append(off)
            omask_all.append(omask)

        coords = np.concatenate(coords_all, 0)
        feats = np.concatenate(feats_all, 0)
        mot = np.concatenate(mot_all, 0)
        sem = np.concatenate(sem_all, 0)
        off = np.concatenate(off_all, 0)
        omask = np.concatenate(omask_all, 0)

        # ---- signed residual-image motion features (fixed width K) ------------
        # Reference points get real residuals vs each past offset; past-frame
        # points get 0 (they exist only for context). K is constant regardless
        # of how many past frames actually loaded (early-in-sequence -> zeros).
        K = self.n_frames - 1
        if self.residual_feats and K > 0:
            past_by_off = {ref - frames[t]: frame_xyz[t]
                           for t in range(1, len(frames))}
            past_list = [past_by_off.get(k, np.zeros((0, 3), np.float32))
                         for k in range(1, self.n_frames)]
            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip)  # (N_ref, K)
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), K), np.float32)
                               for t in range(1, len(frames))]
            residuals = np.concatenate(res_blocks, 0)
            feats = np.concatenate([feats, residuals], axis=1)  # (total, 1+K)
        # -----------------------------------------------------------------------

        # quantize spatial dims to voxel grid; keep t integer as-is
        qcoords = coords.copy()
        qcoords[:, :3] = np.floor(coords[:, :3] / self.voxel_size)
        qcoords = qcoords.astype(np.int32)

        # collapse duplicate (x,y,z,t) voxels -> keep first; aggregate labels by max
        _, uniq = np.unique(qcoords, axis=0, return_index=True)
        uniq.sort()
        return {
            "coords": qcoords[uniq],          # (M, 4) int32  (x, y, z, t)
            "feats": feats[uniq],             # (M, 1+K) float32
            "motion": mot[uniq],              # (M,)  int64
            "semantic": sem[uniq],            # (M,)  int64
            "offset": off[uniq].astype(np.float32),   # (M, 3) float32
            "offset_mask": omask[uniq],       # (M,)  bool
            "meta": (seq, ref),
        }