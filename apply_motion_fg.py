"""Lever E: seed the cluster head from MOTION confidence, not semantic class.

WHY (measured). The cluster-consistency head decides what to cluster using
fg = semantic_argmax in {movable classes}. But semantic mIoU is only ~0.41, so
the completion mechanism -- whose oracle ceiling is 0.808 IoU, i.e. ~+4 over our
0.765 -- is gated by the weakest head in the model. Meanwhile "instances missed
entirely" sits at 22-23% in every run: objects the head never gets a chance to
complete because their voxels were not called foreground.

FIX. Choose the clustering seed set with model.cluster_fg_mode:
  'semantic' : current behaviour (semantic argmax in movable ids)
  'motion'   : voxels whose moving probability exceeds cluster_fg_seed_th
  'union'    : either of the above (widest seed set)
Motion seeding removes the dependency on the semantic head entirely; 'union'
keeps semantic support for objects the motion head is unsure about.

Run once from repo root:  python apply_motion_fg.py   (backs up *.mfbak)
Enable with, e.g.:
  model: { cluster_fg_mode: motion, cluster_fg_seed_th: 0.3 }
"""
import os, shutil
R = os.path.expanduser("~/sparseunet4d")


def patch(path, subs):
    p = os.path.join(R, path)
    s = open(p).read()
    if all(new.split("\n")[0].strip() in s for _, new in subs if new.strip()):
        print(f"skip {path} (looks patched)")
        return
    shutil.copy(p, p + ".mfbak")
    for old, new in subs:
        assert old in s, f"[{path}] pattern not found:\n{old[:90]}"
        s = s.replace(old, new, 1)
    open(p, "w").write(s); print("patched", path)


FG = '''        if self.cluster is not None:
            probm = torch.softmax(motion_logits, 1)[:, 1]
            fg_sem = torch.isin(semantic_logits.argmax(1),
                                self._movable.to(feats.device))
            fg_mot = probm >= self.cluster_fg_seed_th
            if self.cluster_fg_mode == "motion":
                fg = fg_mot
            elif self.cluster_fg_mode == "union":
                fg = fg_sem | fg_mot
            else:
                fg = fg_sem
            fused, clog, rid = self.cluster(feats, coords, motion_logits, fg)'''

# ---- dual_branch.py (the model we actually train) --------------------------
patch("sparseunet4d/models/dual_branch.py", [
    ("""                 cluster_cross_frame=False, cluster_feature_fusion=False):""",
     """                 cluster_cross_frame=False, cluster_feature_fusion=False,
                 cluster_fg_mode="semantic", cluster_fg_seed_th=0.3):"""),
    ("""        self.register_buffer("_movable", torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                             persistent=False)""",
     """        self.cluster_fg_mode = cluster_fg_mode
        self.cluster_fg_seed_th = float(cluster_fg_seed_th)
        self.register_buffer("_movable", torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                             persistent=False)"""),
    ("""        if self.cluster is not None:
            sem_arg = semantic_logits.argmax(1)
            fg = torch.isin(sem_arg, self._movable.to(sem_arg.device))
            fused, clog, rid = self.cluster(feats, coords, motion_logits, fg)""",
     FG),
])

# ---- train.py / eval: forward the two knobs -------------------------------
patch("scripts/train.py", [
    ("""                  cluster_feature_fusion=m.get("cluster_feature_fusion", False))""",
     """                  cluster_feature_fusion=m.get("cluster_feature_fusion", False),
                  cluster_fg_mode=m.get("cluster_fg_mode", "semantic"),
                  cluster_fg_seed_th=m.get("cluster_fg_seed_th", 0.3))"""),
])

print("\nNOTE: SparseUNet4D (single-branch) keeps semantic gating; the knobs are\n"
      "wired for DualBranchUNet4D, which is what dual_v* trains.")
print("Enable with:  model: { cluster_fg_mode: motion, cluster_fg_seed_th: 0.3 }")
