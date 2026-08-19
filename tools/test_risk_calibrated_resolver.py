#!/usr/bin/env python3
"""Regression tests for weighted risk-calibrated resolver utilities."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.train_risk_calibrated_resolver import (
    Cache,
    apply_actions,
    calibrate_actions,
    fit_weighted_binary,
)


features = np.array([
    [2.0, 0.0], [1.5, 0.1], [-1.0, 1.0], [-2.0, 0.9],
], np.float64)
positive = np.array([90, 80, 5, 1])
negative = np.array([10, 20, 95, 99])
model = fit_weighted_binary(features, positive, negative, 0.1)
score = model.predict(features)
assert score[0] > score[-1]
assert np.all((score >= 0) & (score <= 1))

cache = Cache(
    sequence=6, features=features,
    names=np.array(["a", "b"]),
    add_tp=np.array([20, 10, 0, 0]),
    add_fp=np.array([1, 2, 50, 60]),
    keep_tp=np.array([20, 10, 3, 2]),
    keep_fp=np.array([0, 0, 20, 10]),
    map_tp=100, map_fp=40, total_positive=130,
)
add_scores = np.array([0.9, 0.8, 0.1, 0.05])
keep_scores = np.array([0.9, 0.8, 0.1, 0.05])
operating = calibrate_actions(
    cache, add_scores, keep_scores,
    precision_drop=0.05, recall_drop=0.10)
result = operating["result"]
assert result["iou"] > 100 / (100 + 40 + 30)

transferred = apply_actions(
    cache, add_scores, keep_scores,
    operating["add_threshold"], operating["keep_threshold"])
assert transferred["result"] == result

print("Risk-calibrated resolver regression tests passed")
