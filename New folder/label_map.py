"""SemanticKITTI label mapping for joint semantic + MOS supervision.

Raw .label files store uint32 where the lower 16 bits are the semantic id and
the upper 16 bits are the instance id. SemanticKITTI's *moving* variants use
ids 252-259. For MOS we collapse those to a single moving class; everything
else is static. For the 19-class semantic task we use the official
`learning_map` from semantic-kitti.yaml (load it rather than hardcode, so you
stay aligned with the benchmark).

We deliberately keep the moving-class id set hardcoded because it is stable and
well documented, and it defines the binary motion ground truth.
"""

from __future__ import annotations
import numpy as np
import yaml

# Stable across the SemanticKITTI MOS benchmark.
MOVING_IDS = {252, 253, 254, 255, 256, 257, 258, 259}
# 252 moving-car, 253 moving-bicyclist, 254 moving-person,
# 255 moving-motorcyclist, 256 moving-on-rails, 257 moving-bus,
# 258 moving-truck, 259 moving-other-vehicle
IGNORE_INDEX = -1


def load_semantic_learning_map(yaml_path: str):
    """Return (learning_map, learning_map_inv, num_classes) from official yaml."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    learning_map = cfg["learning_map"]
    learning_map_inv = cfg["learning_map_inv"]
    # build a fast lookup table indexed by raw id
    max_id = max(learning_map.keys())
    lut = np.full((max_id + 1,), IGNORE_INDEX, dtype=np.int64)
    for raw, mapped in learning_map.items():
        lut[raw] = mapped
    num_classes = len(set(learning_map_inv.keys()))
    return lut, learning_map_inv, num_classes


def split_label(label: np.ndarray):
    """uint32 .label -> (semantic_id, instance_id)."""
    sem = label & 0xFFFF
    inst = label >> 16
    return sem.astype(np.int64), inst.astype(np.int64)


def to_motion_labels(sem_raw: np.ndarray) -> np.ndarray:
    """Raw semantic ids -> binary motion labels {0 static, 1 moving}."""
    moving = np.isin(sem_raw, list(MOVING_IDS))
    return moving.astype(np.int64)


def to_semantic_labels(sem_raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Raw semantic ids -> learning ids via the official LUT (with ignore)."""
    out = np.full_like(sem_raw, IGNORE_INDEX)
    valid = sem_raw < len(lut)
    out[valid] = lut[sem_raw[valid]]
    return out
