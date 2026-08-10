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
                       normalize=True, clip=None, return_validity=False):
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
    K = len(past_scans_in_now)
    feats = np.zeros((N, K), dtype=np.float32)
    # validity: 1 = past pixel HAD a return (residual is real evidence),
    # 0 = past pixel empty (newly occupied OR just sparse). Without this the
    # two cases are indistinguishable, both encoded as residual 0.
    valid_ch = np.zeros((N, K), dtype=np.float32)

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
        valid_ch[:, k] = has_past.astype(np.float32)
    if return_validity:
        return np.concatenate([feats, valid_ch], axis=1)      # (N, 2K)
    return feats


def temporal_residual_blocks(frame_xyz, stack_offsets, offsets, *, clip=3.0,
                             return_validity=False, all_frames=False):
    """Build residual features for every loaded temporal slice.

    Channel count is always ``len(offsets)`` (or twice that with validity).
    With ``all_frames=False`` this exactly reproduces the legacy layout: real
    residuals on the reference slice and zeros on context slices.  With
    ``all_frames=True``, each slice compares against all other canonical scans;
    the reference block remains bit-identical to legacy behavior.
    """
    if not frame_xyz:
        return []
    offsets = list(offsets)
    stack_offsets = list(stack_offsets)
    if len(frame_xyz) != len(stack_offsets):
        raise ValueError("frame_xyz and stack_offsets must have the same length")
    past_by_off = {stack_offsets[t]: frame_xyz[t]
                   for t in range(1, len(frame_xyz))}
    empty = np.zeros((0, 3), np.float32)
    past_list = [past_by_off.get(o, empty) for o in offsets]
    ref = residual_channels(frame_xyz[0], past_list, normalize=False, clip=clip,
                            return_validity=return_validity)
    if not all_frames:
        width = ref.shape[1]
        return [ref] + [np.zeros((len(frame_xyz[t]), width), np.float32)
                        for t in range(1, len(frame_xyz))]

    scans = {0: frame_xyz[0], **past_by_off}
    canonical = [0] + offsets
    blocks = []
    for t, query_off in enumerate(stack_offsets):
        compare = [scans.get(o, empty) for o in canonical if o != query_off]
        blocks.append(residual_channels(
            frame_xyz[t], compare, normalize=False, clip=clip,
            return_validity=return_validity))
    return blocks
