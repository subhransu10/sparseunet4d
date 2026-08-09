"""Lever D: free-space / newly-occupied channels for the residual features.

MEASURED MOTIVATION. residual_channels() sets the residual to 0 whenever the
past range image has NO return at a point's pixel:

    has_past = isfinite(r_past) & valid_now
    res[has_past] = r_now - r_past          # else stays 0

so "no evidence" and "no motion" are encoded identically. A pedestrian walking
through open space moves away from a pixel that has nothing behind it -> past
pixel empty -> residual 0 -> looks static. A car almost always has building or
ground behind it -> real residual. This predicts exactly what we measure:
moving-person recall stuck at 0.763-0.765 while moving-bicyclist is 0.978, and
extending the temporal baseline 8 -> 16 frames changed person recall by 0.002
(the pixel is still empty, so a longer window cannot help).

FIX. Emit an extra channel per offset: 1 where the past pixel HAD a return
(residual is real evidence), 0 where it was empty (currently-occupied,
previously-unobserved -> possible motion, possible sparsity). The network can
then learn to read "new occupancy" as motion in context instead of having the
distinction destroyed. Input channels go 1+K -> 1+2K.

Enable with  dataset.residual_validity: true
Run once from repo root:  python apply_freespace.py   (backs up *.fsbak)
"""
import os, shutil
R = os.path.expanduser("~/sparseunet4d")


def patch(path, subs):
    p = os.path.join(R, path)
    s = open(p).read(); shutil.copy(p, p + ".fsbak")
    for old, new in subs:
        assert old in s, f"[{path}] pattern not found:\n{old[:90]}"
        s = s.replace(old, new, 1)
    open(p, "w").write(s); print("patched", path)


# ---- 1. residual_features.py: optional validity channels -------------------
patch("sparseunet4d/datasets/residual_features.py", [
    ("""def residual_channels(points_now, past_scans_in_now,
                       H=64, W=2048, fov_up_deg=3.0, fov_down_deg=-25.0,
                       normalize=True, clip=None):""",
     """def residual_channels(points_now, past_scans_in_now,
                       H=64, W=2048, fov_up_deg=3.0, fov_down_deg=-25.0,
                       normalize=True, clip=None, return_validity=False):"""),
    ("""    _, u_now, v_now, r_now, valid_now = spherical_project(
        points_now, H, W, fov_up_deg, fov_down_deg)
    N = points_now.shape[0]
    feats = np.zeros((N, len(past_scans_in_now)), dtype=np.float32)""",
     """    _, u_now, v_now, r_now, valid_now = spherical_project(
        points_now, H, W, fov_up_deg, fov_down_deg)
    N = points_now.shape[0]
    K = len(past_scans_in_now)
    feats = np.zeros((N, K), dtype=np.float32)
    # validity: 1 = past pixel HAD a return (residual is real evidence),
    # 0 = past pixel empty (newly occupied OR just sparse). Without this the
    # two cases are indistinguishable, both encoded as residual 0.
    valid_ch = np.zeros((N, K), dtype=np.float32)"""),
    ("""        if clip is not None:
            res = np.clip(res, -clip, clip)
        feats[:, k] = res.astype(np.float32)
    return feats""",
     """        if clip is not None:
            res = np.clip(res, -clip, clip)
        feats[:, k] = res.astype(np.float32)
        valid_ch[:, k] = has_past.astype(np.float32)
    if return_validity:
        return np.concatenate([feats, valid_ch], axis=1)      # (N, 2K)
    return feats"""),
])

# ---- 2. dataset: flag + wider residual block -------------------------------
patch("sparseunet4d/datasets/semantickitti.py", [
    ("""                 feat_rep="label", all_frame_labels=False,
                 inject_class_boost=None):""",
     """                 feat_rep="label", all_frame_labels=False,
                 inject_class_boost=None, residual_validity=False):"""),
    ("""        self.inject_class_boost = inject_class_boost or {}""",
     """        self.inject_class_boost = inject_class_boost or {}
        # lever D: emit a per-offset "past pixel had a return" channel so the
        # net can distinguish 'no motion' from 'no observation'.
        self.residual_validity = residual_validity"""),
    ("""            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip)  # (N_ref, K)
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), K), np.float32)
                               for t in range(1, len(frame_xyz))]""",
     """            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip,
                                  return_validity=self.residual_validity)
            Kc = R.shape[1]                       # K, or 2K with validity
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), Kc), np.float32)
                               for t in range(1, len(frame_xyz))]"""),
])

# ---- 3. in_ch: 1 + K  ->  1 + 2K when validity is on -----------------------
patch("scripts/train.py", [
    ("""    in_ch = 1 + (n_frames - 1) if residual_feats else 1""",
     """    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)
    in_ch = 1 + _k if residual_feats else 1"""),
    ("""        inject_class_boost=d.get("inject_class_boost"))""",
     """        inject_class_boost=d.get("inject_class_boost"),
        residual_validity=d.get("residual_validity", False))"""),
])

patch("eval_mos_official.py", [
    ("""    in_ch = 1 + (n_frames - 1) if residual_feats else 1""",
     """    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)
    in_ch = 1 + _k if residual_feats else 1"""),
    ("""        feat_rep=d.get("feat_rep", "label"))""",
     """        feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False))"""),
])

print("\nLever D wired. Enable with dataset.residual_validity: true")
print("NOTE: with dual_branch, app_ch stays 1; the motion branch now sees 2K "
      "channels automatically.")
