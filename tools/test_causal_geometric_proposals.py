#!/usr/bin/env python3
"""Regression tests for native causal geometric motion proposals."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.causal_geometric_proposal_eval import (
    causal_range_gaps,
    complete_cluster_scores,
    geometric_disagreement,
    local_range_envelope,
)


image = np.full((5, 8), np.inf)
image[2, 0] = 10.0
low, high, visible = local_range_envelope(image, radius=1)
assert visible[2, 7], "horizontal azimuth must wrap"
assert low[2, 7] == 10.0 and high[2, 7] == 10.0
assert not visible[0, 4], "unobserved rays must remain unobserved"

# Same ray: first point is static, second differs from historical range by 2 m.
current = np.array([[10.0, 0.0, 0.0], [8.0, 0.0, 0.0]], np.float32)
history = np.array([[10.0, 0.0, 0.0]], np.float32)
gaps = causal_range_gaps(current, [history, history], neighborhood_radius=0)
assert np.allclose(gaps[0], 0.0)
assert np.allclose(gaps[1], 2.0)
score, eligible = geometric_disagreement(gaps, 0.5, 2)
assert eligible.tolist() == [True, True]
assert np.allclose(score, [0.0, 1.0])

# No historical return means abstention rather than positive evidence.
empty_gaps = causal_range_gaps(
    np.array([[0.0, 10.0, 0.0]], np.float32),
    [history], neighborhood_radius=0)
score, eligible = geometric_disagreement(empty_gaps, 0.5, 1)
assert not eligible[0] and score[0] == 0.0

points = np.array([0.10, 0.20, 0.05, 0.80], np.float32)
clusters = np.array([4, 4, 4, -1])
geometry = np.array([1.0, 1.0, 0.0, 1.0], np.float32)
eligible = np.array([True, True, True, True])
completed = complete_cluster_scores(
    points, clusters, geometry, eligible,
    boost=0.5, geometric_floor=0.5, minimum_coverage=0.25)
assert np.allclose(completed[:3], 0.70), completed
assert completed[3] == points[3], "unclustered points must not be promoted"

blocked = complete_cluster_scores(
    points, clusters, geometry, np.zeros(4, bool),
    boost=0.5, geometric_floor=0.5, minimum_coverage=0.25)
assert np.allclose(blocked[:3], 0.20), blocked

print("Causal geometric-proposal regression tests passed")
