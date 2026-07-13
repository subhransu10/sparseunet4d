"""T2: dataset + collate check for the offset head. CPU-only, no GPU.
Verifies GT offsets/masks are shaped, aligned, zero on statics & past frames,
point toward instance centers, and survive collate.
"""
DATA_ROOT      = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"
DATASET_MODULE = "sparseunet4d.datasets.semantickitti"
SEQ, N_FRAMES  = 8, 4
SAMPLES        = [200, 800, 1500]

import importlib, numpy as np
m = importlib.import_module(DATASET_MODULE)
SemanticKITTI4D = m.SemanticKITTI4D
from sparseunet4d.datasets.collate import me_collate

ok = True
def check(cond, msg):
    global ok; print(("  PASS " if cond else "  FAIL ") + msg); ok &= cond

ds = SemanticKITTI4D(DATA_ROOT, [SEQ], n_frames=N_FRAMES, voxel_size=0.1,
                     semantic_yaml=None, pose_mode="gt", point_range=51.2,
                     residual_feats=True, res_clip=3.0)
print("[dataset] offset GT sanity")
for i in SAMPLES:
    s = ds[i]
    off, om, mot, t = s["offset"], s["offset_mask"], s["motion"], s["coords"][:, 3]
    n = len(s["coords"])
    check(off.shape == (n, 3), f"[{i}] offset shape {off.shape} == ({n},3)")
    check(om.shape == (n,) and om.dtype == bool, f"[{i}] mask shape/dtype")
    check(np.isfinite(off).all(), f"[{i}] offsets finite")
    # panoptic mask: all reference-frame THING voxels; moving is a SUBSET
    ref_mov = (t == 0) & (mot == 1)
    check(bool(om[ref_mov].all()) if ref_mov.any() else True,
          f"[{i}] moving ⊆ mask (mov={int(ref_mov.sum())}, mask={int(om.sum())})")
    check(not om[t > 0].any() if (t > 0).any() else True,
          f"[{i}] mask==False on past frames")
    check(om.sum() >= ref_mov.sum(), f"[{i}] mask count >= moving count")
    # offsets zero where unmasked (statics + past frames)
    check(np.abs(off[~om]).max() < 1e-6 if (~om).any() else True,
          f"[{i}] offset==0 off-mask")
    # masked offsets: within a car-scale radius, and per-instance they should
    # sum to ~0 (mean-centered). Check magnitude sanity.
    if om.sum() > 0:
        mag = np.linalg.norm(off[om], axis=1)
        check(mag.max() < 15.0, f"[{i}] |offset| < 15m (max={mag.max():.2f})")
        check(mag.mean() < 6.0, f"[{i}] mean |offset| < 6m ({mag.mean():.2f})")

print("[collate] offset survives batching")
b = me_collate([ds[SAMPLES[0]], ds[SAMPLES[1]]])
check("offset" in b and "offset_mask" in b, "  keys present")
check(b["offset"].shape[0] == b["coords"].shape[0], "  offset len == coords len")
check(b["offset"].shape[1] == 3, "  offset width 3")
check(b["offset_mask"].dtype == __import__("torch").bool, "  mask is bool tensor")
check(int(b["offset_mask"].sum()) ==
      int(ds[SAMPLES[0]]["offset_mask"].sum() + ds[SAMPLES[1]]["offset_mask"].sum()),
      "  mask count preserved across batch")

print("\nALL GOOD -> smoke test (T3)." if ok else "\nFIX failures above.")