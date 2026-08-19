#!/usr/bin/env python3
"""Calibrate safe Sparse additions on one recovery regime and transfer them.

This is intentionally add-only: the label-free prevalence gate has already
selected the recovery regime, so MapMOS positives are never removed. Cluster
reliability is learned and calibrated on the source sequence; the operating
point is then applied unchanged to the held-out target sequence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.train_risk_calibrated_resolver import (
    action_curves,
    apply_actions,
    fit_weighted_binary,
    load_cache,
    metrics,
    print_result,
)


def calibrate_add_only(cache, scores, precision_drop):
    thresholds, add_tp, add_fp = action_curves(
        scores, cache.add_tp, cache.add_fp, "add")
    tp = cache.map_tp + add_tp
    fp = cache.map_fp + add_fp
    fn = cache.total_positive - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    baseline = metrics(cache.map_tp, cache.map_fp, cache.total_positive)
    feasible = precision >= baseline["precision"] - float(precision_drop)
    feasible_iou = np.where(feasible, iou, -np.inf)
    best = feasible_iou.max()
    candidates = np.flatnonzero(feasible_iou == best)
    # Prefer the least invasive action when IoU ties.
    changed = add_tp + add_fp
    index = int(candidates[np.argmin(changed[candidates])])
    return {
        "threshold": float(thresholds[index]),
        "result": metrics(int(tp[index]), int(fp[index]),
                          cache.total_positive),
        "added_tp": int(add_tp[index]),
        "added_fp": int(add_fp[index]),
    }


def apply_add_only(cache, scores, threshold):
    # keep_threshold=-inf makes the veto mask empty.
    return apply_actions(cache, scores, np.ones(len(scores)),
                         threshold, -np.inf)


def save_frozen_model(path, weighted_model, feature_names, threshold,
                      train_sequence, regularization_c, precision_drop):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_names=np.asarray(feature_names),
        scaler_mean=np.asarray(weighted_model.scaler.mean_, np.float64),
        scaler_scale=np.asarray(weighted_model.scaler.scale_, np.float64),
        coefficient=np.asarray(weighted_model.model.coef_[0], np.float64),
        intercept=np.asarray(weighted_model.model.intercept_[0], np.float64),
        add_threshold=np.asarray(threshold, np.float64),
        train_sequence=np.asarray(train_sequence, np.int32),
        regularization_c=np.asarray(regularization_c, np.float64),
        precision_drop=np.asarray(precision_drop, np.float64),
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--train-sequence", type=int, default=7)
    parser.add_argument("--test-sequence", type=int, default=8)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--precision-drop", type=float, default=0.005)
    parser.add_argument("--minimum-test-delta", type=float, default=0.005)
    parser.add_argument("--model-output",
                        help="write frozen scaler, classifier and threshold")
    args = parser.parse_args()

    root = Path(args.cache_dir)
    train = load_cache(
        root / f"dual_expert_seq{args.train_sequence:02d}.npz")
    test = load_cache(
        root / f"dual_expert_seq{args.test_sequence:02d}.npz")
    if not np.array_equal(train.names, test.names):
        raise ValueError("cache feature schemas differ")

    model = fit_weighted_binary(
        train.features, train.add_tp, train.add_fp,
        args.regularization_c)
    train_scores = model.predict(train.features)
    operating = calibrate_add_only(
        train, train_scores, args.precision_drop)
    if args.model_output:
        model_path = save_frozen_model(
            args.model_output, model, train.names, operating["threshold"],
            train.sequence, args.regularization_c, args.precision_drop)
        print(f"frozen model: {model_path}")
    test_scores = model.predict(test.features)
    transferred = apply_add_only(
        test, test_scores, operating["threshold"])

    train_baseline = metrics(
        train.map_tp, train.map_fp, train.total_positive)
    test_baseline = metrics(
        test.map_tp, test.map_fp, test.total_positive)
    delta = transferred["result"]["iou"] - test_baseline["iou"]

    print("=== held-out recovery-regime cluster resolver ===")
    print(f"train_sequence={train.sequence:02d} "
          f"test_sequence={test.sequence:02d} features={len(train.names)}")
    print(f"C={args.regularization_c} "
          f"precision_drop<={args.precision_drop:.4f}")
    print(f"frozen add threshold={operating['threshold']:.8g}")
    print_result("train MapMOS", train_baseline)
    print_result("train calibrated", operating["result"])
    print(f"train additions: TP={operating['added_tp']} "
          f"FP={operating['added_fp']}")
    print_result("test MapMOS", test_baseline)
    print_result("test transferred", transferred["result"])
    print(f"test additions: TP={transferred['added_tp']} "
          f"FP={transferred['added_fp']}")
    print(f"held-out IoU delta={delta:+.4f}")
    decision = "ACCEPT" if delta >= args.minimum_test_delta else "REJECT"
    print(f"DECISION: {decision} recovery-cluster resolver")


if __name__ == "__main__":
    main()
