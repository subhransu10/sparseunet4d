"""Sanity checks that run anywhere (no SemanticKITTI, no MinkowskiEngine).

Verifies the parts most likely to be wrong:
  1. SE(3) exp map produces valid rotations.
  2. GT relative-pose registration is exact (identity round-trip).
  3. Drift compounds: error grows with distance from the reference frame.

Run:  python tools/test_poses.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sparseunet4d.datasets.poses import (
    _se3_exp, GTPoseProvider, DriftyPoseProvider,
)


def test_se3_exp_is_rotation():
    xi = np.array([0.1, -0.2, 0.05, 1.0, 2.0, -0.5])
    T = _se3_exp(xi)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "det(R) != 1"
    print("  [ok] se3_exp produces a valid rotation")


class _FakeGT(GTPoseProvider):
    """Build a provider from synthetic poses without reading files."""
    def __init__(self, poses):
        self.poses = poses


def _make_trajectory(n=20):
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(1, n):
        # drive forward 1 m and yaw slightly each step
        step = _se3_exp(np.array([0, 0, 0.02, 1.0, 0, 0]))
        poses[i] = poses[i - 1] @ step
    return poses


def test_gt_registration_exact():
    gt = _FakeGT(_make_trajectory())
    # a point fixed in the world should land identically after src->ref->world
    p_world = np.array([5.0, 1.0, 0.3, 1.0])
    ref, src = 10, 4
    # express world point in src frame, then register src->ref, then ref->world
    p_src = np.linalg.inv(gt.poses[src]) @ p_world
    T = gt.relative(src, ref)
    p_ref = T @ p_src
    p_back = gt.poses[ref] @ p_ref
    assert np.allclose(p_back, p_world, atol=1e-9), "registration not exact"
    print("  [ok] GT relative registration is exact")


def test_drift_compounds():
    gt = _FakeGT(_make_trajectory())
    drift = DriftyPoseProvider(gt, rot_std_deg=0.5, trans_std_m=0.1, seed=0)
    ref = len(gt) - 1
    errs = []
    for offset in [1, 5, 10, 18]:
        src = ref - offset
        T_gt = gt.relative(src, ref)
        T_dr = drift.relative(src, ref)
        # translational disagreement between true and drifted registration
        errs.append(np.linalg.norm((np.linalg.inv(T_gt) @ T_dr)[:3, 3]))
    assert errs[0] < errs[-1], f"drift did not grow with distance: {errs}"
    print(f"  [ok] drift grows with frame distance: "
          f"{[round(e,3) for e in errs]} m")


if __name__ == "__main__":
    print("Running pose sanity checks...")
    test_se3_exp_is_rotation()
    test_gt_registration_exact()
    test_drift_compounds()
    print("All pose checks passed.")
