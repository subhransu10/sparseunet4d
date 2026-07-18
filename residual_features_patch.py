"""APPEND to sparseunet4d/datasets/residual_features.py (keeps existing API).

One projection pass produces BOTH the per-point residual channels (unchanged
numerics vs residual_v2) AND the residual-image stack + per-point pixel ids
for the 2D branch. Avoids projecting twice in __getitem__.
"""
import numpy as np
# (uses spherical_project already defined in this module)


def residual_images_and_channels(points_now, past_scans_in_now,
                                 H=64, W=2048, fov_up_deg=3.0,
                                 fov_down_deg=-25.0, normalize=False,
                                 clip=None, range_norm=51.2):
    """Returns:
      res_pts   : (N, K) signed per-point residuals (identical to
                  residual_channels(normalize, clip))
      imgs      : (2+K, H, W) float32 = [range/range_norm, K residual imgs,
                  valid_frac] -- clipped, 0 where empty
      pixel_idx : (N,) int64 flat v*W+u for valid current points, else -1
    """
    now_img, u, v, r_now, valid = spherical_project(
        points_now, H, W, fov_up_deg, fov_down_deg)
    N = points_now.shape[0]
    K = len(past_scans_in_now)
    res_pts = np.zeros((N, K), np.float32)
    res_imgs = np.zeros((K, H, W), np.float32)
    vcount = np.zeros((H, W), np.float32)

    for k, past in enumerate(past_scans_in_now):
        p_img, *_ = spherical_project(past, H, W, fov_up_deg, fov_down_deg)
        # image residual
        ok = np.isfinite(now_img) & np.isfinite(p_img)
        r = np.zeros((H, W), np.float64)
        r[ok] = now_img[ok] - p_img[ok]
        if normalize:
            r[ok] /= np.maximum(now_img[ok], 1e-3)
        if clip is not None:
            r = np.clip(r, -clip, clip)
        res_imgs[k] = r.astype(np.float32)
        vcount += ok
        # per-point residual (same math as residual_channels)
        rp = p_img[v, u]
        hp = np.isfinite(rp) & valid
        res = np.zeros(N, np.float64)
        res[hp] = r_now[hp] - rp[hp]
        if normalize:
            res[hp] /= np.maximum(r_now[hp], 1e-3)
        if clip is not None:
            res = np.clip(res, -clip, clip)
        res_pts[:, k] = res.astype(np.float32)

    rng_ch = np.where(np.isfinite(now_img), now_img / range_norm, 0.0)
    valid_ch = vcount / max(K, 1)
    imgs = np.concatenate([rng_ch[None].astype(np.float32), res_imgs,
                           valid_ch[None].astype(np.float32)], 0)

    pixel_idx = np.full(N, -1, np.int64)
    pixel_idx[valid] = v[valid] * W + u[valid]
    return res_pts, imgs, pixel_idx