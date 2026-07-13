"""
Residual-image motion features for LiDAR MOS.

Core idea (LiDAR-MOS / MotionSeg3D style): given the current scan and one or
more PAST scans already transformed into the current sensor frame (you have
verified pose registration), compare per-ray range. A static surface lands at
the same range after ego-compensation -> residual ~ 0. A mover lands elsewhere
-> nonzero residual, even when the geometric cue alone is weak.

Output: one SIGNED residual channel per temporal offset, aligned to the
current-frame points. Signed so approaching (neg) vs receding (pos) is encoded.

No MinkowskiEngine / torch needed here -- pure numpy, runs on CPU. This is the
exact function used both by the synthetic validator and the seq-08 separability
eval, so passing the mock validates the real path.
"""
import numpy as np


def spherical_project(points, H=64, W=2048, fov_up_deg=3.0, fov_down_deg=-25.0):
    """
    Project Nx3 points (sensor frame) to a range image via min-pooling.
    Returns:
        range_img : (H, W) float, np.inf where empty
        u, v      : (N,) int pixel cols/rows for each input point
        r         : (N,) float per-point range
        valid     : (N,) bool  (inside image bounds & r>0)
    KITTI HDL-64E defaults: fov_up=3 deg, fov_down=-25 deg.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    valid = r > 1e-6

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_up) + abs(fov_down)

    yaw = np.arctan2(y, x)
    # pitch only where valid; avoid divide-by-zero
    pitch = np.zeros_like(r)
    pitch[valid] = np.arcsin(np.clip(z[valid] / r[valid], -1.0, 1.0))

    u = 0.5 * (1.0 - yaw / np.pi) * W            # [0, W]
    v = (1.0 - (pitch + abs(fov_down)) / fov) * H  # [0, H]

    u = np.floor(u).astype(np.int64)
    v = np.floor(v).astype(np.int64)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    valid &= (v >= 0) & (v < H) & (u >= 0) & (u < W)

    range_img = np.full((H, W), np.inf, dtype=np.float64)
    # min-pool: nearest surface wins. Sort by descending range so smallest
    # range is written last.
    idx = np.where(valid)[0]
    order = idx[np.argsort(-r[idx])]
    range_img[v[order], u[order]] = r[order]
    return range_img, u, v, r, valid


def residual_channels(points_now, past_scans_in_now,
                       H=64, W=2048, fov_up_deg=3.0, fov_down_deg=-25.0,
                       normalize=True, clip=None):
    """
    points_now         : (N,3) current-frame points (sensor frame).
    past_scans_in_now  : list of (M_k,3) past scans ALREADY transformed into the
                         current sensor frame (ego-compensated), ordered by
                         increasing offset (e.g. t-1, t-2, t-4).
    Returns (N, K) signed residual features aligned to points_now.
      residual = r_now - r_past_at_same_pixel
      normalize: divide by r_now (scale-invariant; recommended).
      Empty past pixel -> 0 (no evidence).
    """
    _, u_now, v_now, r_now, valid_now = spherical_project(
        points_now, H, W, fov_up_deg, fov_down_deg)
    N = points_now.shape[0]
    feats = np.zeros((N, len(past_scans_in_now)), dtype=np.float32)

    for k, past in enumerate(past_scans_in_now):
        past_img, _, _, _, _ = spherical_project(
            past, H, W, fov_up_deg, fov_down_deg)
        r_past = past_img[v_now, u_now]           # (N,) per current point
        has_past = np.isfinite(r_past) & valid_now
        res = np.zeros(N, dtype=np.float64)
        res[has_past] = r_now[has_past] - r_past[has_past]
        if normalize:
            res[has_past] /= np.maximum(r_now[has_past], 1e-3)
        if clip is not None:
            res = np.clip(res, -clip, clip)
        feats[:, k] = res.astype(np.float32)
    return feats