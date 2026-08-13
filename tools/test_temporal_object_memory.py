"""Small regression tests for the causal temporal object-memory probe."""
from __future__ import annotations

import numpy as np

from experiments.temporal_object_memory_eval import (
    Proposal,
    TemporalObjectMemory,
    cluster_complete,
    reliability_filter,
)


def proposal(cluster_id, x, score, semantic_class=1):
    return Proposal(
        cluster_id=cluster_id,
        centroid_world=np.array([x, 0.0, 0.0], np.float32),
        radius=0.5,
        semantic_class=semantic_class,
        score=score,
        voxel_count=10,
    )


def main():
    memory = TemporalObjectMemory(alpha=0.9, association_gate=1.0, max_age=2)
    first = memory.update(7, 0, [proposal(0, 0.0, 0.9)])
    assert first[0] == 0.9

    weak = memory.update(7, 1, [proposal(3, 0.2, 0.001)])
    assert weak[3] > 0.001, weak
    assert weak[3] < 0.9, weak
    reliable = reliability_filter(
        weak, memory.last_evidence, association_gate=1.0)
    assert 3 in reliable, reliable

    # A far proposal must start a new track and receive no memory boost.
    far = memory.update(7, 2, [proposal(4, 20.0, 0.002)])
    assert far[4] == 0.002, far
    assert not reliability_filter(
        far, memory.last_evidence, association_gate=1.0), far

    # Changing sequence resets all causal state.
    reset = memory.update(8, 0, [proposal(0, 0.2, 0.003)])
    assert reset[0] == 0.003, reset

    points = np.array([0.001, 0.4, 0.002], np.float32)
    clusters = np.array([0, -1, 0], np.int64)
    completed = cluster_complete(points, clusters, {0: 0.1})
    np.testing.assert_allclose(completed, [0.1, 0.4, 0.1])

    print("Temporal object-memory regression tests passed")


if __name__ == "__main__":
    main()
