"""
Synthetic validator for residual_features. No data, no GPU.
Proves: (1) static surface -> residual ~ 0, (2) radial mover residual scales
linearly with velocity, (3) tangential mover is still flagged.
"""
import numpy as np
from residual_features import residual_channels

rng = np.random.default_rng(0)
DT = 0.1  # 10 Hz


def make_wall(x=10.0, n=4000, spread=8.0):
    y = rng.uniform(-spread, spread, n)
    z = rng.uniform(-2.0, 2.0, n)
    x = np.full(n, x) + rng.normal(0, 0.02, n)
    return np.stack([x, y, z], 1)


def make_blob(center, n=400, r=0.4):
    return center + rng.normal(0, r, (n, 3))


def scene(mover_offset):
    """Static wall + one mover. Returns (now, past) with past = now - motion."""
    wall = make_wall()
    mover_now = make_blob(np.array([5.0, 0.0, 0.0]))
    mover_past = mover_now - mover_offset          # where it was one frame ago
    now = np.vstack([wall, mover_now])
    past = np.vstack([wall, mover_past])           # wall identical (static ego+scene)
    is_moving = np.concatenate([np.zeros(len(wall)), np.ones(len(mover_now))])
    return now, past, is_moving.astype(bool)


print("velocity(m/s)  dir         median|res| STATIC   median|res| MOVER")
for v, direction, off in [
    (0.0,  "none",       np.array([0, 0, 0.0])),
    (2.0,  "radial",     np.array([2.0, 0, 0]) * DT),   # away along +x
    (5.0,  "radial",     np.array([5.0, 0, 0]) * DT),
    (10.0, "radial",     np.array([10.0, 0, 0]) * DT),
    (5.0,  "tangential", np.array([0, 5.0, 0]) * DT),   # across rays
]:
    now, past, mov = scene(off)
    feats = residual_channels(now, [past], normalize=False)  # raw meters
    res = np.abs(feats[:, 0])
    s = np.median(res[~mov])
    m = np.median(res[mov])
    print(f"  {v:5.1f}       {direction:11s}  {s:14.4f}   {m:14.4f}")

print("\nExpected: STATIC ~0 always; radial MOVER ~= v*dt (0.2/0.5/1.0 m); "
      "tangential still nonzero via appearance/disappearance.")