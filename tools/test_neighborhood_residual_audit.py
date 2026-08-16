#!/usr/bin/env python3
"""Regression tests for fixed neighborhood residual representation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.multisequence_neighborhood_residual_audit import (
    neighborhood_residual,
    representative_rows,
)


query = np.asarray([[10.0, 0.0, 0.0]], np.float32)
# Half of one 2048-column angular bin moves yaw=0 into the adjacent column
# after the projection's floor operation, while remaining inside radius=1.
yaw = np.pi / 2048.0
past = np.asarray(
    [[10.0 * np.cos(yaw), 10.0 * np.sin(yaw), 0.0]], np.float32
)

same_pixel_value, same_pixel_valid = neighborhood_residual(
    query, past, radius_pixels=0
)
assert not same_pixel_valid[0]
assert same_pixel_value[0] == 0.0

neighbor_value, neighbor_valid = neighborhood_residual(
    query, past, radius_pixels=1
)
assert neighbor_valid[0]
assert abs(float(neighbor_value[0])) < 1e-4

point_voxel = np.asarray([4, 4, 9], np.int64)
residual = np.asarray([[0.1, 0.2], [0.8, 0.1], [0.3, 0.4]], np.float32)
rows = representative_rows(point_voxel, residual)
assert set(rows.tolist()) == {1, 2}

print("Neighborhood-residual regression tests passed")
