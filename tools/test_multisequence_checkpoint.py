#!/usr/bin/env python3
"""Regression test for robust multi-sequence checkpoint selection."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparseunet4d.utils.metrics import (
    MovingThresholdMeter,
    best_shared_worst,
)


def meter(tp, fp, fn):
    result = MovingThresholdMeter([0.2, 0.8])
    result.tp = torch.as_tensor(tp, dtype=torch.float64).clone()
    result.fp = torch.as_tensor(fp, dtype=torch.float64).clone()
    result.fn = torch.as_tensor(fn, dtype=torch.float64).clone()
    return result


seq06 = meter([70, 60], [10, 10], [20, 30])
seq07 = meter([40, 65], [20, 10], [40, 25])
pooled = meter(
    seq06.tp + seq07.tp,
    seq06.fp + seq07.fp,
    seq06.fn + seq07.fn,
)

best = best_shared_worst({6: seq06, 7: seq07}, pooled)
assert best["threshold"] == 0.8
assert abs(best["iou"] - 0.6) < 1e-12
assert abs(best["sequence_ious"][6] - 0.6) < 1e-12
assert abs(best["sequence_ious"][7] - 0.65) < 1e-12
assert abs(best["pooled_iou"] - 0.625) < 1e-12
assert abs(best["macro_iou"] - 0.625) < 1e-12
print("Multi-sequence checkpoint regression test passed")
