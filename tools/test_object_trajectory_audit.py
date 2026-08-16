#!/usr/bin/env python3
"""Regression tests for the frozen object-trajectory audit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))

from experiments.multisequence_object_trajectory_audit import (
    MatchObservation,
    ObjectDescriptor,
    associate_objects,
    trajectory_features,
)


def descriptor(cluster_id, xyz, semantic_class=1, radius=1.0):
    return ObjectDescriptor(
        cluster_id=cluster_id,
        centroid=np.asarray(xyz, np.float32),
        radius=radius,
        semantic_class=semantic_class,
        voxel_count=20,
    )


# One-to-one same-class matching must recover two valid associations.
reference = [
    descriptor(10, [4.0, 0.0, 0.0]),
    descriptor(11, [0.0, 5.0, 0.0], semantic_class=2),
]
past = [
    descriptor(20, [2.0, 0.0, 0.0]),
    descriptor(21, [0.0, 5.1, 0.0], semantic_class=2),
]
matches = associate_objects(
    reference,
    past,
    offset_scans=2,
    scan_period_seconds=0.1,
    maximum_speed_mps=20.0,
    base_gate_m=2.0,
    maximum_size_ratio=2.0,
)
assert set(matches) == {10, 11}
np.testing.assert_allclose(matches[10].displacement, [2.0, 0.0, 0.0])
np.testing.assert_allclose(
    matches[11].displacement,
    [0.0, -0.1, 0.0],
    rtol=1e-6,
    atol=1e-7,
)

# A semantic mismatch must not be associated even when the centroids coincide.
assert associate_objects(
    [descriptor(1, [0.0, 0.0, 0.0], semantic_class=1)],
    [descriptor(2, [0.0, 0.0, 0.0], semantic_class=2)],
    offset_scans=1,
    scan_period_seconds=0.1,
    maximum_speed_mps=20.0,
    base_gate_m=2.0,
    maximum_size_ratio=2.0,
) == {}

# A constant 10 m/s trajectory should be coherent, monotonic and fully tracked.
observations = [
    MatchObservation(1, np.asarray([1.0, 0.0, 0.0]), 1.0),
    MatchObservation(2, np.asarray([2.0, 0.0, 0.0]), 1.0),
    MatchObservation(4, np.asarray([4.0, 0.0, 0.0]), 1.0),
]
features = trajectory_features(observations, 3, 0.1)
assert abs(features["maximum_displacement"] - 4.0) < 1e-12
assert abs(features["furthest_displacement"] - 4.0) < 1e-12
assert abs(features["fitted_speed"] - 10.0) < 1e-12
assert abs(features["median_speed"] - 10.0) < 1e-12
assert abs(features["direction_consistency"] - 1.0) < 1e-12
assert abs(features["speed_consistency"] - 1.0) < 1e-12
assert abs(features["monotonicity"] - 1.0) < 1e-12
assert abs(features["track_support"] - 1.0) < 1e-12
assert abs(features["displacement_support"] - 4.0) < 1e-12
assert abs(features["trajectory_score"] - 10.0) < 1e-12

# No matches must produce a neutral all-zero feature vector.
empty = trajectory_features([], 3, 0.1)
assert all(value == 0.0 for value in empty.values())

print("Object-trajectory audit regression tests passed")
