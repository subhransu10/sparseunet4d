"""
Wiring check for residual-channel integration. CPU-only, no GPU, no training.
Verifies the dataset produces correctly-shaped/aligned residual features and
that the model's stem expects the matching in_ch. Run on the box that has the
data; it never touches the GPU.
"""
DATA_ROOT      = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"  # <-- same as eval
DATASET_MODULE = "sparseunet4d.datasets.semantickitti"
SEQ, N_FRAMES  = 8, 4
SAMPLES        = [200, 800, 1500]   # mid-sequence indices (past frames exist)

import importlib, numpy as np
m = importlib.import_module(DATASET_MODULE)
SemanticKITTI4D = m.SemanticKITTI4D
from sparseunet4d.models.model import SparseUNet4D

K = N_FRAMES - 1
ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg); ok &= cond

print(f"[dataset] residual_feats=True, n_frames={N_FRAMES}  (expect feat width {1+K})")
ds = SemanticKITTI4D(DATA_ROOT, [SEQ], n_frames=N_FRAMES, voxel_size=0.1,
                     semantic_yaml=None, pose_mode="gt", point_range=51.2,
                     residual_feats=True, res_clip=3.0)
for i in SAMPLES:
    s = ds[i]
    c, f, mot, t = s["coords"], s["feats"], s["motion"], s["coords"][:, 3]
    n = len(c)
    check(f.shape[1] == 1 + K, f"[{i}] feat width {f.shape[1]} == {1+K}")
    check(len(f) == n == len(mot), f"[{i}] coords/feats/motion aligned ({n})")
    check(np.isfinite(f).all(), f"[{i}] no NaN/Inf in feats")
    check(np.abs(f[:, 1:]).max() <= 3.0 + 1e-4, f"[{i}] residuals within clip +/-3")
    # residual channels must be ~0 on past-frame points (t>0)
    past = t > 0
    if past.any():
        check(np.abs(f[past, 1:]).max() < 1e-6, f"[{i}] residuals==0 on past frames")
    # and nonzero on at least some reference movers (t==0, motion==1)
    mov = (t == 0) & (mot == 1)
    if mov.sum() > 0:
        frac = np.mean(np.abs(f[mov, 1:]).max(1) > 0.02)
        check(frac > 0.1, f"[{i}] residual fires on movers (frac={frac:.2f}, movers={int(mov.sum())})")

print("[dataset] residual_feats=False  (expect feat width 1, baseline A/B)")
ds0 = SemanticKITTI4D(DATA_ROOT, [SEQ], n_frames=N_FRAMES, voxel_size=0.1,
                      semantic_yaml=None, pose_mode="gt", point_range=51.2,
                      residual_feats=False)
check(ds0[SAMPLES[0]]["feats"].shape[1] == 1, "  width==1 when disabled")

print(f"[model] SparseUNet4D(in_ch={1+K}) stem expects {1+K} channels")
net = SparseUNet4D(in_ch=1 + K, num_semantic=20, base=16, n_stages=2)
check(net.stem.conv1.in_ch == 1 + K, f"  stem.conv1.in_ch == {1+K}")

print("\nALL GOOD -> proceed to 5k sanity run." if ok else "\nFIX failures above first.")