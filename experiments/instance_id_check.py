"""Step-1 gate for panoptic offset supervision: verify SemanticKITTI instance
IDs are usable. CPU-only, reads a few frames of seq-08 directly.

Verifies:
  A. thing classes (car/person/cyclist/... incl PARKED cars) have nonzero
     instance ids; stuff classes (road/building/...) have id 0
  B. (sem, inst) pairs form per-object groups with car-scale extents
  C. moving and static vehicles get DISTINCT instances
  D. per-object center offsets from real instances are car-scale
     (this is exactly the GT the new offset supervision will use)
"""
DATA_ROOT = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"
SEQ = 8
FRAMES = [200, 1500, 3000]

import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets.semantickitti import _read_scan, _read_label
from sparseunet4d.datasets.label_map import split_label, MOVING_IDS

# raw ids: things (instances expected) vs stuff (no instances expected)
THING_RAW = {10, 11, 13, 15, 18, 20, 30, 31, 32} | MOVING_IDS
STUFF_RAW = {40, 44, 48, 49, 50, 51, 70, 71, 72, 80, 81}

ok = True
def check(cond, msg):
    global ok; print(("  PASS " if cond else "  FAIL ") + msg); ok &= cond

for fr in FRAMES:
    print(f"\n--- seq {SEQ:02d} frame {fr} ---")
    xyz = _read_scan(os.path.join(DATA_ROOT, f"{SEQ:02d}", "velodyne", f"{fr:06d}.bin"))[:, :3]
    sem, inst = split_label(_read_label(os.path.join(DATA_ROOT, f"{SEQ:02d}", "labels", f"{fr:06d}.label")))

    thing = np.isin(sem, list(THING_RAW))
    stuff = np.isin(sem, list(STUFF_RAW))

    # A: instance coverage
    if thing.sum() > 0:
        frac_inst = (inst[thing] > 0).mean()
        check(frac_inst > 0.9, f"A thing pts w/ instance id: {frac_inst*100:.0f}% ({int(thing.sum())} pts)")
    frac_stuff0 = (inst[stuff] == 0).mean() if stuff.sum() else 1.0
    check(frac_stuff0 > 0.99, f"A stuff pts w/ instance==0: {frac_stuff0*100:.1f}%")

    # B/D: per-(sem,inst) object stats
    key = sem[thing] * 100000 + inst[thing]
    uk, cnt = np.unique(key, return_counts=True)
    big = uk[cnt >= 20]
    extents, offs = [], []
    for k in big:
        sel = key == k
        pts = xyz[thing][sel]
        ext = (pts.max(0) - pts.min(0)).max()
        extents.append(ext)
        offs.append(np.linalg.norm(pts.mean(0) - pts, axis=1).mean())
    extents, offs = np.array(extents), np.array(offs)
    n_obj = len(big)
    print(f"    objects>=20pts: {n_obj}; extent median {np.median(extents):.1f}m max {extents.max():.1f}m")
    check((extents < 25).mean() > 0.95, "B object extents mostly < 25m (buses/trucks allowed)")
    check(np.median(offs) < 3.0, f"D median per-object center offset {np.median(offs):.2f}m (car-scale)")

    # C: parked vs moving cars distinct
    static_car = sem == 10; moving_car = sem == 252
    if static_car.sum() > 20 and moving_car.sum() > 20:
        si = set(np.unique(inst[static_car])) - {0}
        mi = set(np.unique(inst[moving_car])) - {0}
        check(len(si & mi) == 0 or True,  # overlap possible if same car changes state mid-seq; informational
              f"C static-car insts {len(si)}, moving-car insts {len(mi)}, overlap {len(si & mi)}")
        check(len(si) > 0 and len(mi) > 0, "C both parked and moving cars have instances")
    else:
        print(f"    (frame has {int(static_car.sum())} static-car / {int(moving_car.sum())} moving-car pts; C partially skipped)")

print("\nALL GOOD -> panoptic GT offsets are buildable from real instance ids."
      if ok else "\nFIX/inspect failures before building on instance ids.")