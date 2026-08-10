"""CPU-only regression tests for the score-improvement infrastructure."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sparseunet4d.datasets.residual_features import (
    residual_channels,
    temporal_residual_blocks,
)
from sparseunet4d.utils.metrics import MovingThresholdMeter
from sparseunet4d.models.losses import instance_detection_loss


def _scan(moving_range):
    # Static ray plus one object moving along a second ray.  Keeping direction
    # fixed guarantees a shared spherical pixel and a measurable range delta.
    return np.asarray([
        [10.0, 0.0, 0.0],
        [moving_range, moving_range * 0.2, 0.0],
    ], dtype=np.float32)


def test_dense_temporal_residuals():
    frames = [_scan(r) for r in (5.0, 4.8, 4.5, 4.0, 3.0)]
    stack_offsets = [0, 1, 2, 4, 8]
    offsets = [1, 2, 4, 8]

    legacy = temporal_residual_blocks(
        frames, stack_offsets, offsets, clip=3.0, all_frames=False)
    dense = temporal_residual_blocks(
        frames, stack_offsets, offsets, clip=3.0, all_frames=True)

    direct = residual_channels(frames[0], frames[1:], normalize=False, clip=3.0)
    np.testing.assert_array_equal(legacy[0], direct)
    np.testing.assert_array_equal(dense[0], direct)
    assert all(np.count_nonzero(x) == 0 for x in legacy[1:])
    assert any(np.count_nonzero(x) > 0 for x in dense[1:])
    assert all(x.shape == (2, 4) for x in dense)

    dense_valid = temporal_residual_blocks(
        frames, stack_offsets, offsets, clip=3.0,
        return_validity=True, all_frames=True)
    assert all(x.shape == (2, 8) for x in dense_valid)


def test_threshold_grid_covers_high_confidence_models():
    # At threshold 0.5 this gives one FP; at 0.9 it is perfect.
    probs = torch.tensor([0.99, 0.95, 0.80, 0.10])
    logits = torch.stack([torch.log1p(-probs), torch.log(probs)], dim=1)
    labels = torch.tensor([1, 1, 0, 0])
    meter = MovingThresholdMeter()
    meter.update(logits, labels)
    best = meter.best()
    assert best["threshold"] > 0.5
    assert best["iou"] == 1.0
    assert best["iou_argmax"] < best["iou"]


def test_instance_detection_loss_reaches_every_object():
    # Object 0 has many voxels, object 1 has one. Both must receive gradient.
    logits = torch.zeros(5, 2, requires_grad=True)
    instance = torch.tensor([0, 0, 0, 0, 1])
    loss = instance_detection_loss(logits, instance, topk_fraction=0.1)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0:4].abs().sum() > 0
    assert logits.grad[4].abs().sum() > 0


if __name__ == "__main__":
    test_dense_temporal_residuals()
    test_threshold_grid_covers_high_confidence_models()
    test_instance_detection_loss_reaches_every_object()
    print("SOTA improvement regression tests passed")
