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
                 residual_feats=True, res_clip=3.0,
                 return_point_map=False, frame_offsets=None,
                 augment=False, aug_scale=(0.95, 1.05), aug_rot_deg=180.0,
                 fixed_transform=None,
                 inject_bank=None, inject_prob=0.0, inject_max_n=4):
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
        self.return_point_map = return_point_map
        # Temporal offsets into the past (t index 0 == reference, offset 0).
        # Strided offsets (e.g. [1, 2, 4, 8]) widen the window so slow / nearby
        # movers displace enough between scans to leave a residual signal.
        # Default (None) reproduces consecutive frames [1 .. n_frames-1].
        self.offsets = (list(frame_offsets) if frame_offsets
                        else list(range(1, n_frames)))
        self.n_frames = 1 + len(self.offsets)   # authoritative frame count
        # train-time geometric augmentation (identical transform for every frame
        # in the stack, so registration and the residual signal are preserved).
        self.augment = augment
        self.aug_scale = aug_scale
        self.aug_rot_deg = aug_rot_deg
        # deterministic per-view transform for test-time augmentation; when set
        # it overrides random `augment`. Use range-preserving transforms (D4) so
        # the box range-clip keeps the same points in the same order across views.
        self.fixed_transform = (None if fixed_transform is None
                                else np.asarray(fixed_transform, np.float32))
        # trajectory-consistent moving-instance injection (train only): paste
        # real movers from the bank with ONE rigid transform applied to all of
        # their frames, so the residual channels see a genuine motion signature.
        self.inject_bank_path = inject_bank
        self.inject_prob = float(inject_prob)
        self.inject_max_n = int(inject_max_n)
        self._bank = None   # lazy per-worker load

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

    def _ensure_bank(self):
        if self._bank is None:
            self._bank = np.load(self.inject_bank_path, allow_pickle=True)
        return self._bank

    def _inject_movers(self, stack_offsets, frame_xyz, coords_all, feats_all,
                       mot_all, sem_all, off_all, omask_all):
        """Paste 1..inject_max_n bank movers into this stack.

        One rigid yaw+translation per instance is applied to EVERY frame of its
        trajectory, so its ego-compensated displacement (what the residual
        channels measure) is preserved exactly. Placement is rejected if the
        target area is already occupied in the reference frame. Must run BEFORE
        residual computation so injected points get real residuals.
        """
        rng = np.random.default_rng()
        if rng.random() > self.inject_prob or len(frame_xyz) < 2:
            return
        bank = self._ensure_bank()
        ref_xy = frame_xyz[0][:, :2]
        for _ in range(int(rng.integers(1, self.inject_max_n + 1))):
            inst = bank[int(rng.integers(len(bank)))]
            pts0 = inst["frames"][0]
            c0 = pts0[:, :3].mean(0)
            placed = None
            for _try in range(5):
                th = rng.uniform(-np.pi, np.pi)
                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
                r = rng.uniform(6.0, 35.0); ang = rng.uniform(-np.pi, np.pi)
                target = np.array([r * np.cos(ang), r * np.sin(ang)], np.float32)
                # translation that moves the (rotated) ref centroid to target;
                # z untouched: bank and scene share the sensor's mounting height.
                t = np.zeros(3, np.float32)
                t[:2] = target - (R[:2, :2] @ c0[:2].astype(np.float32))
                p0 = pts0[:, :3] @ R.T + t
                if self.point_range is not None and \
                        np.abs(p0[:, :2]).max() >= self.point_range - 1.0:
                    continue
                # free-space check: few existing ref points inside the footprint
                lo, hi = p0[:, :2].min(0) - 0.3, p0[:, :2].max(0) + 0.3
                occ = ((ref_xy > lo) & (ref_xy < hi)).all(1).sum()
                if occ > 0.10 * len(p0) + 20:
                    continue
                placed = (R, t)
                break
            if placed is None:
                continue
            R, t = placed
            sem_lab = (int(to_semantic_labels(
                np.array([inst["sem_raw"]]), self.lut)[0])
                if self.lut is not None else IGNORE_INDEX)
            for t_idx, o in enumerate(stack_offsets):
                if o not in inst["frames"]:
                    continue        # object absent that far back (newly visible)
                arr = inst["frames"][o]
                xyz = (arr[:, :3] @ R.T + t).astype(np.float32)
                rem = arr[:, 3:4].astype(np.float32)
                if self.point_range is not None:
                    m = np.all(np.abs(xyz) < self.point_range, axis=1)
                    xyz, rem = xyz[m], rem[m]
                if len(xyz) == 0:
                    continue
                n = len(xyz)
                frame_xyz[t_idx] = np.concatenate([frame_xyz[t_idx], xyz], 0)
                t_col = np.full((n, 1), t_idx, dtype=np.float32)
                coords_all[t_idx] = np.concatenate(
                    [coords_all[t_idx], np.concatenate([xyz, t_col], 1)], 0)
                feats_all[t_idx] = np.concatenate([feats_all[t_idx], rem], 0)
                if t_idx == 0:
                    mot_i = np.ones(n, np.int64)
                    sem_i = np.full(n, sem_lab, np.int64)
                    off_i = (xyz.mean(0) - xyz).astype(np.float32)
                    om_i = np.ones(n, bool)
                else:
                    mot_i = np.full(n, IGNORE_INDEX, np.int64)
                    sem_i = np.full(n, IGNORE_INDEX, np.int64)
                    off_i = np.zeros((n, 3), np.float32)
                    om_i = np.zeros(n, bool)
                mot_all[t_idx] = np.concatenate([mot_all[t_idx], mot_i], 0)
                sem_all[t_idx] = np.concatenate([sem_all[t_idx], sem_i], 0)
                off_all[t_idx] = np.concatenate([off_all[t_idx], off_i], 0)
                omask_all[t_idx] = np.concatenate([omask_all[t_idx], om_i], 0)

    def _aug_matrix(self):
        """Random z-rotation + x/y flips + uniform scale as one 3x3 matrix.
        Applied identically to every frame in the stack; a fresh entropy-seeded
        RNG avoids the numpy-in-DataLoader-workers duplication pitfall."""
        rng = np.random.default_rng()
        th = np.deg2rad(rng.uniform(-self.aug_rot_deg, self.aug_rot_deg))
        c, s = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
        fx = -1.0 if rng.random() < 0.5 else 1.0
        fy = -1.0 if rng.random() < 0.5 else 1.0
        Fl = np.diag([fx, fy, 1.0]).astype(np.float32)
        sc = float(rng.uniform(self.aug_scale[0], self.aug_scale[1]))
        return (sc * (Rz @ Fl)).astype(np.float32)

    def __getitem__(self, i):
        seq, ref = self.index[i]
        provider = self.pose_providers[seq]
        # temporal stack: (t_idx, frame, offset). t_idx 0 == reference (offset 0).
        # Early-in-sequence frames whose offset runs before frame 0 are dropped;
        # their residual channels become zeros (handled below), keeping K fixed.
        stack = [(t, ref - o, o) for t, o in enumerate([0] + self.offsets)
                 if ref - o >= 0]
        stack_offsets = [o for (_, _, o) in stack]
        # one shared transform for the whole stack: a fixed TTA view if given,
        # else a random train-time augmentation, else none.
        if self.fixed_transform is not None:
            aug_R = self.fixed_transform
        elif self.augment:
            aug_R = self._aug_matrix()
        else:
            aug_R = None

        coords_all, feats_all, mot_all, sem_all = [], [], [], []
        off_all, omask_all = [], []
        frame_xyz = []  # keep clipped xyz per frame for residual computation
        for t_idx, f, o in stack:
            bin_p, lab_p = self._frame_paths(seq, f)
            scan = _read_scan(bin_p)
            xyz, remission = scan[:, :3], scan[:, 3:4]

            # register past frame into reference frame using (possibly drifted) pose
            if t_idx != 0:
                T = provider.relative(f, ref)
                xyz = _transform(xyz, T).astype(np.float32)

            # augment AFTER registration so the same transform hits every frame
            # (relative alignment preserved -> static surfaces still cancel).
            if aug_R is not None:
                xyz = (xyz @ aug_R.T).astype(np.float32)

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

        # trajectory-consistent mover injection (train only) — before residuals
        # so injected points participate in the range-image comparison.
        if self.inject_bank_path is not None and self.inject_prob > 0:
            self._inject_movers(stack_offsets, frame_xyz, coords_all, feats_all,
                                mot_all, sem_all, off_all, omask_all)

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
            # map temporal offset -> that past frame's xyz (skip the reference)
            past_by_off = {stack_offsets[t]: frame_xyz[t]
                           for t in range(1, len(frame_xyz))}
            # channel k corresponds to self.offsets[k]; a missing offset (early
            # in the sequence) contributes an empty past -> zero residual.
            past_list = [past_by_off.get(o, np.zeros((0, 3), np.float32))
                         for o in self.offsets]
            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip)  # (N_ref, K)
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), K), np.float32)
                               for t in range(1, len(frame_xyz))]
            residuals = np.concatenate(res_blocks, 0)
            feats = np.concatenate([feats, residuals], axis=1)  # (total, 1+K)
        # -----------------------------------------------------------------------

        # quantize spatial dims to voxel grid; keep t integer as-is
        qcoords = coords.copy()
        qcoords[:, :3] = np.floor(coords[:, :3] / self.voxel_size)
        qcoords = qcoords.astype(np.int32)

        # collapse duplicate (x,y,z,t) voxels. Each voxel takes ALL its labels
        # from a single representative point chosen by motion priority
        # (moving > static > ignore), so a voxel is MOVING whenever ANY of its
        # points is moving. Previously we kept the first point and silently
        # dropped moving labels at mixed voxels -> a bias against the moving
        # class in both the GT and the training signal.
        uniq_coords, inv = np.unique(qcoords, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        G = uniq_coords.shape[0]
        rep = np.empty(G, dtype=np.int64)
        order = np.argsort(mot, kind="stable")   # ascending: ignore(-1),static(0),moving(1)
        rep[inv[order]] = order                  # last write per voxel = max-priority point
        out = {
            "coords": uniq_coords,                    # (M, 4) int32  (x, y, z, t)
            "feats": feats[rep],                      # (M, 1+K) float32
            "motion": mot[rep],                       # (M,)  int64
            "semantic": sem[rep],                     # (M,)  int64
            "offset": off[rep].astype(np.float32),    # (M, 3) float32
            "offset_mask": omask[rep],                # (M,)  bool
            "meta": (seq, ref),
        }
        if self.return_point_map:
            # full-resolution reference-frame labels + the voxel row each point
            # maps to, so eval can propagate voxel predictions back to EVERY
            # point (official point-level MOS protocol) rather than scoring the
            # deduplicated voxels. Reference frame is block 0 of the stack.
            n_ref = len(frame_xyz[0])
            out["ref_point_motion"] = mot[:n_ref].astype(np.int64)
            out["ref_point_voxel"] = inv[:n_ref].astype(np.int64)
        return out