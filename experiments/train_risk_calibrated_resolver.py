#!/usr/bin/env python3
"""Leave-one-sequence-out evaluation of a risk-calibrated expert resolver.

Two weighted logistic models estimate object-level action reliability:
  * ADD: precision of SparseUNet cluster points that MapMOS calls static.
  * KEEP: precision of MapMOS-positive points in the cluster; low confidence
    permits a SparseUNet-informed veto.

Each cluster row carries exact point counts, so calibration and evaluation use
the official point metric without expanding the compact cache. Model fitting
and action-threshold selection use only the training sequence in each fold.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class Cache:
    sequence: int
    features: np.ndarray
    names: np.ndarray
    add_tp: np.ndarray
    add_fp: np.ndarray
    keep_tp: np.ndarray
    keep_fp: np.ndarray
    map_tp: int
    map_fp: int
    total_positive: int


@dataclass
class WeightedBinaryModel:
    scaler: StandardScaler
    model: LogisticRegression

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(
            self.scaler.transform(features))[:, 1]


def load_cache(path: str | Path) -> Cache:
    data = np.load(path)
    return Cache(
        sequence=int(data["sequence"]),
        features=data["features"].astype(np.float64),
        names=data["feature_names"],
        add_tp=data["add_tp"].astype(np.int64),
        add_fp=data["add_fp"].astype(np.int64),
        keep_tp=data["keep_tp"].astype(np.int64),
        keep_fp=data["keep_fp"].astype(np.int64),
        map_tp=int(data["map_tp"]),
        map_fp=int(data["map_fp"]),
        total_positive=int(data["total_positive"]),
    )


def fit_weighted_binary(features: np.ndarray, positive: np.ndarray,
                        negative: np.ndarray, regularization_c: float):
    """Fit point-weighted logistic regression from aggregate object rows."""
    total = positive + negative
    usable = total > 0
    x = np.asarray(features[usable], np.float64)
    pos = np.asarray(positive[usable], np.float64)
    neg = np.asarray(negative[usable], np.float64)
    if pos.sum() <= 0 or neg.sum() <= 0:
        raise ValueError("both weighted classes must be present")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    duplicated_x = np.concatenate([scaled, scaled], axis=0)
    labels = np.concatenate([
        np.ones(len(x), np.int64), np.zeros(len(x), np.int64)])
    weights = np.concatenate([pos, neg])
    weights *= len(weights) / max(weights.sum(), 1.0)
    model = LogisticRegression(
        C=float(regularization_c), solver="lbfgs", max_iter=2000,
        random_state=2701)
    model.fit(duplicated_x, labels, sample_weight=weights)
    return WeightedBinaryModel(scaler, model)


def metrics(tp: int, fp: int, total_positive: int) -> dict:
    fn = int(total_positive) - int(tp)
    return {
        "tp": int(tp), "fp": int(fp), "fn": fn,
        "iou": int(tp) / max(int(tp) + int(fp) + fn, 1),
        "precision": int(tp) / max(int(tp) + int(fp), 1),
        "recall": int(tp) / max(int(total_positive), 1),
    }


def candidate_thresholds(scores: np.ndarray, mode: str,
                         count: int = 101) -> np.ndarray:
    finite = np.asarray(scores[np.isfinite(scores)], np.float64)
    if not len(finite):
        raise ValueError("empty score array")
    values = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, count)))
    if mode == "add":
        # +inf = add nothing; -inf = add everything.
        return np.r_[-np.inf, values, np.inf]
    if mode == "veto":
        # -inf = veto nothing; +inf = veto everything.
        return np.r_[-np.inf, values, np.inf]
    raise ValueError(mode)


def action_curves(scores: np.ndarray, positive: np.ndarray,
                  negative: np.ndarray, mode: str):
    thresholds = candidate_thresholds(scores, mode)
    selected_positive = np.zeros(len(thresholds), np.int64)
    selected_negative = np.zeros(len(thresholds), np.int64)
    for index, threshold in enumerate(thresholds):
        selected = scores >= threshold if mode == "add" else scores < threshold
        selected_positive[index] = int(positive[selected].sum())
        selected_negative[index] = int(negative[selected].sum())
    return thresholds, selected_positive, selected_negative


def calibrate_actions(cache: Cache, add_scores: np.ndarray,
                      keep_scores: np.ndarray, precision_drop: float,
                      recall_drop: float):
    """Select joint add/veto thresholds under point-level risk constraints."""
    add_th, add_tp, add_fp = action_curves(
        add_scores, cache.add_tp, cache.add_fp, "add")
    veto_th, lost_tp, removed_fp = action_curves(
        keep_scores, cache.keep_tp, cache.keep_fp, "veto")
    tp = cache.map_tp + add_tp[:, None] - lost_tp[None, :]
    fp = cache.map_fp + add_fp[:, None] - removed_fp[None, :]
    fn = cache.total_positive - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(cache.total_positive, 1)
    baseline = metrics(cache.map_tp, cache.map_fp, cache.total_positive)
    feasible = (
        (precision >= baseline["precision"] - float(precision_drop))
        & (recall >= baseline["recall"] - float(recall_drop))
        & (tp >= 0) & (fp >= 0)
    )
    if not feasible.any():
        raise RuntimeError("no resolver operating point satisfies risk limits")
    feasible_iou = np.where(feasible, iou, -np.inf)
    best_value = feasible_iou.max()
    candidates = np.argwhere(feasible_iou == best_value)
    # Deterministic tie-break: change the fewest points.
    changed = ((add_tp + add_fp)[:, None]
               + (lost_tp + removed_fp)[None, :])
    choice = candidates[np.argmin(changed[tuple(candidates.T)])]
    add_index, veto_index = (int(value) for value in choice)
    return {
        "add_threshold": float(add_th[add_index]),
        "keep_threshold": float(veto_th[veto_index]),
        "result": metrics(
            int(tp[add_index, veto_index]),
            int(fp[add_index, veto_index]), cache.total_positive),
        "added_tp": int(add_tp[add_index]),
        "added_fp": int(add_fp[add_index]),
        "lost_tp": int(lost_tp[veto_index]),
        "removed_fp": int(removed_fp[veto_index]),
    }


def apply_actions(cache: Cache, add_scores: np.ndarray,
                  keep_scores: np.ndarray, add_threshold: float,
                  keep_threshold: float):
    add = add_scores >= add_threshold
    veto = keep_scores < keep_threshold
    added_tp, added_fp = int(cache.add_tp[add].sum()), int(cache.add_fp[add].sum())
    lost_tp = int(cache.keep_tp[veto].sum())
    removed_fp = int(cache.keep_fp[veto].sum())
    result = metrics(
        cache.map_tp + added_tp - lost_tp,
        cache.map_fp + added_fp - removed_fp,
        cache.total_positive)
    return {
        "result": result, "added_tp": added_tp, "added_fp": added_fp,
        "lost_tp": lost_tp, "removed_fp": removed_fp,
    }


def print_result(prefix: str, result: dict):
    print(f"{prefix:<22} IoU={result['iou']:.4f} "
          f"P={result['precision']:.4f} R={result['recall']:.4f} "
          f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--precision-drop", type=float, default=0.005)
    parser.add_argument("--recall-drop", type=float, default=0.02)
    args = parser.parse_args()
    if len(args.sequences) != 2:
        raise ValueError("this audit requires exactly two development sequences")
    caches = {
        sequence: load_cache(
            Path(args.cache_dir) / f"dual_expert_seq{sequence:02d}.npz")
        for sequence in args.sequences
    }
    first_names = caches[args.sequences[0]].names
    if not np.array_equal(first_names, caches[args.sequences[1]].names):
        raise ValueError("cache feature schemas differ")

    fold_deltas = []
    print("=== leave-one-sequence-out risk-calibrated resolver ===")
    print(f"features={len(first_names)} C={args.regularization_c} "
          f"precision_drop<={args.precision_drop:.4f} "
          f"recall_drop<={args.recall_drop:.4f}")
    for train_sequence, test_sequence in (
            (args.sequences[0], args.sequences[1]),
            (args.sequences[1], args.sequences[0])):
        train, test = caches[train_sequence], caches[test_sequence]
        add_model = fit_weighted_binary(
            train.features, train.add_tp, train.add_fp,
            args.regularization_c)
        keep_model = fit_weighted_binary(
            train.features, train.keep_tp, train.keep_fp,
            args.regularization_c)
        train_add = add_model.predict(train.features)
        train_keep = keep_model.predict(train.features)
        operating = calibrate_actions(
            train, train_add, train_keep,
            args.precision_drop, args.recall_drop)
        test_add = add_model.predict(test.features)
        test_keep = keep_model.predict(test.features)
        transferred = apply_actions(
            test, test_add, test_keep,
            operating["add_threshold"], operating["keep_threshold"])
        baseline_train = metrics(
            train.map_tp, train.map_fp, train.total_positive)
        baseline_test = metrics(
            test.map_tp, test.map_fp, test.total_positive)
        delta = transferred["result"]["iou"] - baseline_test["iou"]
        fold_deltas.append(delta)

        print(f"\ntrain seq {train_sequence:02d} -> test seq {test_sequence:02d}")
        print(f"thresholds: add>={operating['add_threshold']:.6g} "
              f"keep>={operating['keep_threshold']:.6g}")
        print_result("train MapMOS", baseline_train)
        print_result("train calibrated", operating["result"])
        print(f"train actions: +TP={operating['added_tp']} "
              f"+FP={operating['added_fp']} -TP={operating['lost_tp']} "
              f"-FP={operating['removed_fp']}")
        print_result("test MapMOS", baseline_test)
        print_result("test transferred", transferred["result"])
        print(f"test actions: +TP={transferred['added_tp']} "
              f"+FP={transferred['added_fp']} -TP={transferred['lost_tp']} "
              f"-FP={transferred['removed_fp']}")
        print(f"held-out IoU delta: {delta:+.4f}")

    minimum_delta = min(fold_deltas)
    mean_delta = float(np.mean(fold_deltas))
    print(f"\nheld-out minimum delta={minimum_delta:+.4f} "
          f"mean delta={mean_delta:+.4f}")
    decision = "ACCEPT" if minimum_delta >= 0.003 else "REJECT"
    print(f"DECISION: {decision} learned reliability resolver")


if __name__ == "__main__":
    main()
