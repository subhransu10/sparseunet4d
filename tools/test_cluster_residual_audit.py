#!/usr/bin/env python3
"""Regression tests for cluster residual-separability statistics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.multisequence_cluster_residual_audit import (
    auroc,
    cluster_residual_features,
    top_fraction,
)


labels = np.asarray([False, False, True, True])
assert abs(auroc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-12
assert abs(auroc(labels, np.ones(4)) - 0.5) < 1e-12
assert abs(top_fraction(np.asarray([1.0, 2.0, 3.0, 4.0])) - 4.0) < 1e-12

residual = np.asarray(
    [[1.0, -0.2], [1.0, -0.3], [1.0, 0.0], [1.0, 0.0]],
    np.float32,
)
features, saturation = cluster_residual_features(
    residual, median_range=10.0, signal_floor=0.1, clip_value=3.0
)
assert abs(features["residual_coherent"] - 1.0) < 1e-7
assert abs(features["residual_coherent_normalized"] - 0.1) < 1e-7
assert abs(features["residual_sign_consistency"] - 1.0) < 1e-7
assert abs(features["residual_offset_support"] - 1.0) < 1e-7
assert saturation == 0.0

print("Cluster residual-separability regression tests passed")
