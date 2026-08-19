#!/usr/bin/env python3
"""Regression test for add-only recovery calibration."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.recovery_cluster_resolver_eval import (
    apply_add_only,
    calibrate_add_only,
)
from experiments.apply_frozen_recovery_resolver import frozen_scores
from experiments.train_risk_calibrated_resolver import Cache


cache = Cache(
    sequence=7,
    features=np.arange(8, dtype=np.float64).reshape(4, 2),
    names=np.array(["a", "b"]),
    add_tp=np.array([20, 10, 0, 0]),
    add_fp=np.array([1, 2, 50, 60]),
    keep_tp=np.zeros(4, np.int64),
    keep_fp=np.zeros(4, np.int64),
    map_tp=100,
    map_fp=10,
    total_positive=140,
)
scores = np.array([0.9, 0.8, 0.1, 0.05])
operating = calibrate_add_only(cache, scores, precision_drop=0.05)
assert operating["result"]["iou"] > 100 / 150
applied = apply_add_only(cache, scores, operating["threshold"])
assert applied["result"] == operating["result"]
assert applied["lost_tp"] == 0
assert applied["removed_fp"] == 0


class Frozen:
    arrays = {
        "scaler_mean": np.array([0.0, 0.0]),
        "scaler_scale": np.array([1.0, 2.0]),
        "coefficient": np.array([1.0, -1.0]),
        "intercept": np.array(0.25),
    }

    def __getitem__(self, key):
        return self.arrays[key]


frozen = frozen_scores(np.array([[1.0, 2.0], [-1.0, 0.0]]), Frozen())
assert frozen[0] > frozen[1]
assert np.all((frozen >= 0) & (frozen <= 1))

print("Recovery-cluster resolver regression tests passed")
