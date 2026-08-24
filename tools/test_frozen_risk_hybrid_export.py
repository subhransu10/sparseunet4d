#!/usr/bin/env python3
"""Regression test for frozen hybrid test-label fusion logic."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.export_frozen_risk_hybrid_test import fuse_mask


mapmos = np.array([True, True, False, False, False])
cluster = np.array([0, 1, 1, 2, -1])
selected = np.array([False, True, True])
keep = np.array([True, False, True, True, False])

assert np.array_equal(
    fuse_mask("mapmos-abstain", mapmos, cluster, selected, keep),
    mapmos)
assert np.array_equal(
    fuse_mask("conservative-veto", mapmos, cluster, selected, keep),
    np.array([True, False, False, False, False]))
assert np.array_equal(
    fuse_mask("recovery-add", mapmos, cluster, selected, keep),
    np.array([True, True, True, True, False]))

print("Frozen risk-hybrid export regression tests passed")
