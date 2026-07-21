"""Pose handling for SemanticKITTI 4D stacking.

This module is the heart of the ego-motion contribution. A `PoseProvider`
returns, for any frame index in a sequence, the 4x4 SE(3) pose of the
*velodyne* sensor expressed in the world frame (frame-0 frame). The dataset
uses these poses to register past scans into the reference frame.

Two providers are implemented:
  - GTPoseProvider:    the ground-truth SuMa poses shipped with KITTI odometry.
  - DriftyPoseProvider: wraps a base provider and injects *compounding* SE(3)
                        drift so that frames further from the reference are
                        more misaligned -- the realistic failure mode that
                        breaks registration-dependent 4D MOS.

KITTI convention notes (these are the usual gotchas):
  - poses.txt stores a 3x4 row-major matrix per line: the pose of the *camera*
    in the world frame.
  - calib.txt 'Tr:' maps velodyne -> camera coordinates.
  - To get the velodyne pose in world:  T_velo_world = inv(Tr) @ T_cam @ Tr
"""

from __future__ import annotations
import os
import numpy as np


def _read_calib_Tr(calib_path: str) -> np.ndarray:
    """Read the velodyne->camera transform 'Tr' as a 4x4 matrix."""
    Tr = None
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("Tr:") or line.startswith("Tr "):
                vals = [float(v) for v in line.strip().split()[1:]]
                Tr = np.array(vals, dtype=np.float64).reshape(3, 4)
                break
    if Tr is None:
        raise ValueError(f"'Tr' not found in {calib_path}")
    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3, :4] = Tr
    return Tr4


def _read_poses(poses_path: str) -> np.ndarray:
    """Read poses.txt -> (F, 4, 4) camera-in-world matrices."""
    raw = np.loadtxt(poses_path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    F = raw.shape[0]
    poses = np.tile(np.eye(4, dtype=np.float64), (F, 1, 1))
    poses[:, :3, :4] = raw.reshape(F, 3, 4)
    return poses


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3)->SE(3). xi = [rx, ry, rz, tx, ty, tz] (rot in rad)."""
    w = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]], dtype=np.float64)
    if theta < 1e-9:
        R = np.eye(3) + W
        V = np.eye(3)
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta ** 2)
        C = (1 - A) / (theta ** 2)
        R = np.eye(3) + A * W + B * (W @ W)
        V = np.eye(3) + B * W + C * (W @ W)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


class GTPoseProvider:
    """Ground-truth velodyne-in-world poses for one sequence."""

    def __init__(self, seq_dir: str):
        Tr = _read_calib_Tr(os.path.join(seq_dir, "calib.txt"))
        Tr_inv = np.linalg.inv(Tr)
        cam_poses = _read_poses(os.path.join(seq_dir, "poses.txt"))
        # velodyne pose in world = inv(Tr) @ T_cam @ Tr
        self.poses = np.einsum("ij,fjk,kl->fil", Tr_inv, cam_poses, Tr)

    def __len__(self):
        return len(self.poses)

    def pose(self, frame_idx: int) -> np.ndarray:
        return self.poses[frame_idx]

    def relative(self, src_idx: int, ref_idx: int) -> np.ndarray:
        """Transform that maps a point in `src` frame into `ref` frame."""
        return np.linalg.inv(self.poses[ref_idx]) @ self.poses[src_idx]


class DriftyPoseProvider:
    """Wraps a base provider and injects compounding odometry drift.

    We model online-odometry error as a random walk in se(3): each frame adds a
    small incremental error to its predecessor, so the *accumulated* error grows
    with distance from frame 0 -- mirroring real drift. `rot_std`/`trans_std`
    are per-frame increments (deg, m). A fixed `seed` makes evaluation
    deterministic per sequence so robustness curves are reproducible.
    """

    def __init__(self, base: GTPoseProvider, rot_std_deg: float,
                 trans_std_m: float, seed: int = 0):
        self.base = base
        self.seed = int(seed)
        self.rot_std = np.deg2rad(rot_std_deg)   # radians, per-frame-step
        self.trans_std = float(trans_std_m)      # metres, per-frame-step
        self.poses = base.poses                  # unused; relative() is the model

    def __len__(self):
        return len(self.poses)

    def pose(self, frame_idx: int) -> np.ndarray:
        return self.poses[frame_idx]

    def relative(self, src_idx, ref_idx):
        T = self.base.relative(src_idx, ref_idx)          # true relative
        k = abs(ref_idx - src_idx)                        # frames apart
        rng = np.random.default_rng((self.seed, ref_idx, src_idx))
        xi = np.concatenate([
            rng.normal(0.0, self.rot_std * np.sqrt(k), 3),   # walk over k steps
            rng.normal(0.0, self.trans_std * np.sqrt(k), 3),
        ])
        return T @ _se3_exp(xi)          # error in the SOURCE sensor frame


class FilePoseProvider:
    """Poses from a KITTI-format poses file (12 floats per row, 3x4 row-major).

    Covers two paper-critical sources with one class:
      - real odometry (e.g. KISS-ICP output)  -> IoU under REAL pose error
      - pre-generated drifted trajectories (make_drifted_poses.py) -> the SAME
        corrupted poses can be fed to external baselines (4DMOS), so ours and
        theirs are compared under IDENTICAL noise.

    frame='camera' (KITTI convention: poses of the left camera; pass calib to
    convert to velodyne, exactly like GTPoseProvider) or frame='velodyne'
    (file already in the sensor frame; no conversion).
    """

    def __init__(self, poses_path: str, frame: str = "camera",
                 calib_path: str | None = None):
        raw = _read_poses(poses_path)
        if frame == "camera":
            assert calib_path, "camera-frame pose file needs calib.txt (Tr)"
            Tr = _read_calib_Tr(calib_path)
            Tr_inv = np.linalg.inv(Tr)
            self.poses = np.einsum("ij,fjk,kl->fil", Tr_inv, raw, Tr)
        elif frame == "velodyne":
            self.poses = raw
        else:
            raise ValueError(f"unknown frame: {frame}")

    def __len__(self):
        return len(self.poses)

    def pose(self, frame_idx: int) -> np.ndarray:
        return self.poses[frame_idx]

    def relative(self, src_idx: int, ref_idx: int) -> np.ndarray:
        return np.linalg.inv(self.poses[ref_idx]) @ self.poses[src_idx]


def build_pose_provider(seq_dir: str, mode: str = "gt",
                        rot_std_deg: float = 0.0, trans_std_m: float = 0.0,
                        seed: int = 0):
    gt = GTPoseProvider(seq_dir)
    if mode == "gt":
        return gt
    if mode == "drift":
        return DriftyPoseProvider(gt, rot_std_deg, trans_std_m, seed)
    raise ValueError(f"unknown pose mode: {mode}")
