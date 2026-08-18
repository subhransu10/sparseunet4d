#!/usr/bin/env python3
"""Regression tests for label-free multi-point voxel aggregation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.datasets.semantickitti import aggregate_voxel_features

inv = np.array([0, 0, 1, 1, 1], dtype=np.int64)
feats = np.array([
    [0.2, 0.1, 0.9], [0.6, -0.8, 0.2],
    [0.1, -0.2, 0.1], [0.4, 0.7, -0.6], [0.7, -0.3, 0.4],
], dtype=np.float32)
expected = np.array([[0.4, -0.8, 0.9], [0.4, 0.7, -0.6]], np.float32)
got = aggregate_voxel_features(inv, 2, feats)
np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)

perm = np.array([4, 0, 3, 1, 2])
np.testing.assert_allclose(
    aggregate_voxel_features(inv[perm], 2, feats[perm]),
    expected, rtol=0, atol=1e-7,
)
print("Voxel evidence aggregation regression tests passed")
