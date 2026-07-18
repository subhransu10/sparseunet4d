"""
Separability gate for residual-image features on SemanticKITTI val seq-08.
NO training, CPU-only. Reuses your SemanticKITTI4D dataset internals for
loading + pose registration (GT poses), so it matches your real pipeline.

GREEN LIGHT: AUC on the FAR (>=40m) and SPARSE strata >= ~0.60.
Those are the movers your recall misses. If the raw residual separates them,
the sparse backbone will exploit it -> train. If not -> pivot to a learned
range-image branch instead of burning a GPU-day.

-------------------------  SET THESE TWO  -----------------------------------
"""
DATA_ROOT      = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"   # <-- path to .../sequences
DATASET_MODULE = "sparseunet4d.datasets.semantickitti"       # <-- import path of the file you pasted
#                 e.g. if it's dataset/semantickitti.py -> "dataset.semantickitti"
# -----------------------------------------------------------------------------

SEQ      = 8
OFFSETS  = [1, 2, 4]          # frames back -> residual channels
STRIDE   = 20                 # subsample seq-08 for a fast gate (lower = more frames)
POINT_RANGE = 51.2

import importlib
import numpy as np
from sklearn.metrics import roc_auc_score
from residual_features import residual_channels

m = importlib.import_module(DATASET_MODULE)
SemanticKITTI4D = m.SemanticKITTI4D
_read_scan, _read_label, _transform = m._read_scan, m._read_label, m._transform
split_label, to_motion_labels, IGNORE_INDEX = \
    m.split_label, m.to_motion_labels, m.IGNORE_INDEX

# build dataset once: gives us pose providers + frame paths for SEQ
ds = SemanticKITTI4D(DATA_ROOT, [SEQ], n_frames=max(OFFSETS) + 1,
                     voxel_size=0.1, semantic_yaml=None, pose_mode="gt",
                     point_range=POINT_RANGE)
provider = ds.pose_providers[SEQ]
N_FRAMES = len([f for _, f in ds.index if _ == SEQ])


def _clip(xyz, extra=None):
    if POINT_RANGE is None:
        return xyz, extra
    keep = np.all(np.abs(xyz) < POINT_RANGE, axis=1)
    return xyz[keep], (extra[keep] if extra is not None else None)


def reference(ref):
    """Reference-frame points (sensor coords) + motion labels (0/1, -1=ignore)."""
    bin_p, lab_p = ds._frame_paths(SEQ, ref)
    xyz = _read_scan(bin_p)[:, :3].astype(np.float32)
    sem_raw, _ = split_label(_read_label(lab_p))
    mot = to_motion_labels(sem_raw).astype(np.int64)
    xyz, mot = _clip(xyz, mot)
    return xyz, mot


def past_in_current(ref, offset):
    """Frame ref-offset registered into the reference sensor frame."""
    f = ref - offset
    if f < 0:
        return np.zeros((0, 3), np.float32)
    bin_p, _ = ds._frame_paths(SEQ, f)
    xyz = _read_scan(bin_p)[:, :3].astype(np.float32)
    xyz = _transform(xyz, provider.relative(f, ref)).astype(np.float32)
    xyz, _ = _clip(xyz)
    return xyz


def local_density(pts, voxel=0.5):
    q = np.floor(pts / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv]


def main():
    frames = range(max(OFFSETS), N_FRAMES, STRIDE)
    score, y, rng, den = [], [], [], []
    for ref in frames:
        pts, mot = reference(ref)
        past = [past_in_current(ref, o) for o in OFFSETS]
        f = residual_channels(pts, past, normalize=True)      # (N,K) signed
        score.append(np.abs(f).max(1))
        y.append(mot)
        rng.append(np.linalg.norm(pts, axis=1))
        den.append(local_density(pts))

    score = np.concatenate(score)
    y = np.concatenate(y)
    rng = np.concatenate(rng)
    den = np.concatenate(den)

    valid = y != IGNORE_INDEX          # drop ignore points
    score, y, rng, den = score[valid], y[valid].astype(int), rng[valid], den[valid]

    def auc(mask, name):
        yy, ss = y[mask], score[mask]
        if yy.sum() < 10 or (yy == 0).sum() < 10:
            print(f"  {name:24s}  n/a (movers={int(yy.sum())})"); return
        print(f"  {name:24s}  AUC={roc_auc_score(yy, ss):.3f}  "
              f"movers={int(yy.sum()):>7d}  P(mov)={yy.mean():.4f}")

    print(f"\nseq-{SEQ:02d}  frames={len(list(frames))}  offsets={OFFSETS}")
    print("overall:");        auc(np.ones_like(y, bool), "ALL")
    print("by range:")
    auc(rng < 20, "near (<20m)")
    auc((rng >= 20) & (rng < 40), "mid (20-40m)")
    auc(rng >= 40, "FAR (>=40m)   <-- key")
    print("by density:")
    med = np.median(den)
    auc(den >= med, "dense")
    auc(den < med, "SPARSE        <-- key")
    print("\nGO if FAR and SPARSE AUC >= ~0.60.")


if __name__ == "__main__":
    main()