#!/usr/bin/env python3
"""Regression tests for precision-controlled temporal memory."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.multisequence_precision_memory_eval import (
    PriorEvidence,
    Proposal,
    StrictCausalTracker,
    controlled_cluster_scores,
)


def proposal(score=0.10, semantic_class=1, x=0.0, radius=1.0):
    return Proposal(
        cluster_id=0,
        centroid_world=np.asarray([x, 0.0, 0.0], np.float32),
        radius=radius,
        semantic_class=semantic_class,
        score=score,
    )


current = [proposal()]
eligible = {
    0: PriorEvidence(True, 0.80, 2, 0.2, 1.1, True),
}
score = controlled_cluster_scores(
    current, eligible, memory_weight=0.50, evidence_floor=0.05,
    history_threshold=0.20, minimum_history_hits=2, maximum_boost=0.15,
)[0]
assert abs(score - 0.25) < 1e-7, score

# The rule is monotonic but all gates are mandatory.
for evidence, floor, history_threshold, hits in [
    ({0: PriorEvidence(False, 0.80, 2, 0.2, 1.1, True)}, 0.05, 0.20, 2),
    (eligible, 0.20, 0.20, 2),
    ({0: PriorEvidence(True, 0.10, 2, 0.2, 1.1, True)}, 0.05, 0.20, 2),
    ({0: PriorEvidence(True, 0.80, 1, 0.2, 1.1, True)}, 0.05, 0.20, 2),
]:
    result = controlled_cluster_scores(
        current, evidence, memory_weight=0.50, evidence_floor=floor,
        history_threshold=history_threshold, minimum_history_hits=hits,
        maximum_boost=0.15,
    )[0]
    assert abs(result - 0.10) < 1e-7, result

# Association is causal, strict, class-consistent and size-consistent.
tracker = StrictCausalTracker(
    association_gate=0.75,
    max_age=1,
    max_size_ratio=1.5,
    velocity_alpha=0.5,
    history_alpha=0.5,
)
first = tracker.update(6, 0, [proposal(score=0.8)])
assert not first[0].matched
second = tracker.update(6, 1, [proposal(score=0.1, x=0.2)])
assert second[0].matched
assert abs(second[0].prior_score - 0.8) < 1e-7
assert second[0].prior_hits == 1
third = tracker.update(6, 2, [proposal(score=0.1, semantic_class=2, x=0.3)])
assert not third[0].matched

print("Precision-controlled temporal-memory regression tests passed")
