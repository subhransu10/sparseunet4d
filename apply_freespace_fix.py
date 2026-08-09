"""Resume lever D after apply_freespace.py failed partway.

residual_features.py was already patched successfully; this finishes the rest,
anchoring on whatever your files actually contain (with or without the
inject_class_boost patch from lever C). Idempotent: already-applied steps are
skipped rather than erroring.

Run from repo root:  python apply_freespace_fix.py
"""
import os, shutil
R = os.path.expanduser("~/sparseunet4d")


def edit(path, subs, marker):
    """Apply subs unless `marker` already present. Each sub = (old, new, required)."""
    p = os.path.join(R, path)
    s = open(p).read()
    if marker in s:
        print(f"skip {path} (already patched)")
        return
    shutil.copy(p, p + ".fsbak2")
    for old, new, required in subs:
        if old in s:
            s = s.replace(old, new, 1)
        elif required:
            raise SystemExit(f"[{path}] required pattern missing:\n{old[:100]}")
    open(p, "w").write(s)
    print("patched", path)


# ---- residual_features.py: verify step 1 landed ---------------------------
rf = open(os.path.join(R, "sparseunet4d/datasets/residual_features.py")).read()
if "return_validity" not in rf:
    raise SystemExit("residual_features.py is NOT patched - rerun apply_freespace.py first")
print("residual_features.py: ok (return_validity present)")

# ---- dataset --------------------------------------------------------------
edit("sparseunet4d/datasets/semantickitti.py", [
    # constructor: handle both variants (with / without inject_class_boost)
    ('                 feat_rep="label", all_frame_labels=False,\n'
     '                 inject_class_boost=None):',
     '                 feat_rep="label", all_frame_labels=False,\n'
     '                 inject_class_boost=None, residual_validity=False):', False),
    ('                 feat_rep="label", all_frame_labels=False):',
     '                 feat_rep="label", all_frame_labels=False,\n'
     '                 residual_validity=False):', False),
    # store the flag right after the P3 flag
    ("        self.all_frame_labels = all_frame_labels",
     "        self.all_frame_labels = all_frame_labels\n"
     "        # lever D: per-offset 'past pixel had a return' channel, so the net\n"
     "        # can tell 'no motion' from 'no observation'.\n"
     "        self.residual_validity = residual_validity", True),
    # widen the residual block
    ("""            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip)  # (N_ref, K)
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), K), np.float32)
                               for t in range(1, len(frame_xyz))]""",
     """            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip,
                                  return_validity=self.residual_validity)
            Kc = R.shape[1]                       # K, or 2K with validity
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), Kc), np.float32)
                               for t in range(1, len(frame_xyz))]""", True),
], marker="self.residual_validity = residual_validity")

# ---- train.py -------------------------------------------------------------
edit("scripts/train.py", [
    ("    in_ch = 1 + (n_frames - 1) if residual_feats else 1",
     '    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)\n'
     "    in_ch = 1 + _k if residual_feats else 1", True),
    ('        inject_class_boost=d.get("inject_class_boost"))',
     '        inject_class_boost=d.get("inject_class_boost"),\n'
     '        residual_validity=d.get("residual_validity", False))', False),
    ('        all_frame_labels=d.get("all_frame_labels", False))   # P3: train only',
     '        all_frame_labels=d.get("all_frame_labels", False),   # P3: train only\n'
     '        residual_validity=d.get("residual_validity", False))', False),
], marker='residual_validity=d.get("residual_validity"')

# ---- eval_mos_official.py -------------------------------------------------
edit("eval_mos_official.py", [
    ("    in_ch = 1 + (n_frames - 1) if residual_feats else 1",
     '    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)\n'
     "    in_ch = 1 + _k if residual_feats else 1", True),
    ('        feat_rep=d.get("feat_rep", "label"))',
     '        feat_rep=d.get("feat_rep", "label"),\n'
     '        residual_validity=d.get("residual_validity", False))', True),
], marker='residual_validity=d.get("residual_validity"')

print("\nLever D complete. Enable with dataset.residual_validity: true")
