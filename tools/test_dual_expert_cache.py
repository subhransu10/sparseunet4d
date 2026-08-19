#!/usr/bin/env python3
"""Regression tests for the compact dual-expert disagreement cache."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.cache_dual_expert_disagreements import (
    action_outcomes,
    cluster_feature_vector,
    feature_names,
)


xyz = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], np.float32)
sparse = np.array([0.1, 0.7, 0.9], np.float32)
mapmos = np.array([False, True, False])
features = np.array([[1, -2], [3, 4], [-5, 6]], np.float32)
semantic = np.array([2, 2, 7], np.int64)
row = cluster_feature_vector(
    xyz, sparse, mapmos, features, semantic, learned_probability=0.8)
names = feature_names(2)
assert row.shape == (len(names),)
assert np.isfinite(row).all()
assert abs(row[names.index("mapmos_positive_fraction")] - 1 / 3) < 1e-6
assert abs(row[names.index("semantic_fraction_2")] - 2 / 3) < 1e-6
assert abs(row[names.index("semantic_fraction_7")] - 1 / 3) < 1e-6

gt = np.array([True, True, False, False])
mp = np.array([False, True, False, True])
valid = np.array([True, True, True, False])
assert action_outcomes(mp, gt, valid) == (1, 1, 1, 0)

print("Dual-expert disagreement-cache regression tests passed")
