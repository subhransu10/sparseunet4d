#!/usr/bin/env python3
"""Apply a frozen recovery-cluster resolver to a cached target sequence."""
from __future__ import annotations

import argparse

import numpy as np

from experiments.recovery_cluster_resolver_eval import apply_add_only
from experiments.train_risk_calibrated_resolver import (
    load_cache,
    metrics,
    print_result,
)


def frozen_scores(features, model):
    scale = np.asarray(model["scaler_scale"], np.float64)
    if np.any(scale <= 0):
        raise ValueError("invalid frozen feature scale")
    standardized = (
        np.asarray(features, np.float64)
        - np.asarray(model["scaler_mean"], np.float64)
    ) / scale
    logits = (
        standardized @ np.asarray(model["coefficient"], np.float64)
        + float(model["intercept"])
    )
    # Numerically stable sigmoid; clipping does not alter useful probabilities.
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.005)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    model = np.load(args.model)
    if not np.array_equal(cache.names, model["feature_names"]):
        raise ValueError("frozen model and cache feature schemas differ")
    threshold = float(model["add_threshold"])
    scores = frozen_scores(cache.features, model)
    result = apply_add_only(cache, scores, threshold)
    baseline = metrics(cache.map_tp, cache.map_fp, cache.total_positive)
    delta = result["result"]["iou"] - baseline["iou"]

    print("=== frozen recovery-cluster resolver ===")
    print(f"trained_on_sequence={int(model['train_sequence']):02d} "
          f"evaluated_on_sequence={cache.sequence:02d}")
    print(f"add_threshold={threshold:.8g}")
    print_result("MapMOS", baseline)
    print_result("frozen resolver", result["result"])
    print(f"additions: TP={result['added_tp']} FP={result['added_fp']}")
    print(f"held-out IoU delta={delta:+.4f}")
    decision = "PASS" if delta >= args.minimum_delta else "FAIL"
    print(f"FROZEN REPRODUCTION: {decision}")


if __name__ == "__main__":
    main()
