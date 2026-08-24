#!/usr/bin/env python3
"""Evaluate the frozen label-free regime-gated MapMOS/SparseUNet hybrid.

The regime decision uses only the raw fraction of MapMOS moving predictions.
Calibration sequences 06 and 07 define a log-space midpoint. A fixed margin
around that midpoint causes the SparseUNet expert to abstain. Ground truth is
used only by the delegated official-metric evaluator after the action and its
threshold have been frozen.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


CALIBRATION_LOW = 0.00099123
CALIBRATION_HIGH = 0.00749227
PREVALENCE_GATE = math.sqrt(CALIBRATION_LOW * CALIBRATION_HIGH)
DEFAULT_MARGIN_FACTOR = 1.10
VETO_THRESHOLD = 0.001
ADD_THRESHOLD = 0.995


def raw_motion_prevalence(directory: str | Path) -> tuple[float, int, int, int]:
    paths = sorted(Path(directory).glob("*.label"))
    if not paths:
        raise FileNotFoundError(f"no .label predictions in {directory}")
    total = moving = 0
    for path in paths:
        semantic = np.fromfile(path, dtype=np.uint32) & 0xFFFF
        total += len(semantic)
        moving += int(((semantic >= 251) & (semantic <= 259)).sum())
    if total == 0:
        raise ValueError("prediction set is empty")
    return moving / total, moving, total, len(paths)


def choose_regime(prevalence: float, gate: float = PREVALENCE_GATE,
                  margin_factor: float = DEFAULT_MARGIN_FACTOR) -> str:
    if gate <= 0 or margin_factor < 1:
        raise ValueError("gate must be positive and margin factor must be >= 1")
    if prevalence < gate / margin_factor:
        return "conservative-veto"
    if prevalence > gate * margin_factor:
        return "recovery-add"
    return "mapmos-abstain"


def selected_row(output: str, mode: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(mode)}\s+\d", re.MULTILINE)
    match = pattern.search(output)
    if match is None:
        raise RuntimeError(f"could not find {mode!r} result row")
    return output[match.start():output.find("\n", match.start())].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mapmos-pred-dir", required=True)
    parser.add_argument("--pool", default="top25")
    parser.add_argument("--margin-factor", type=float,
                        default=DEFAULT_MARGIN_FACTOR)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    prevalence, moving, total, frames = raw_motion_prevalence(
        args.mapmos_pred_dir)
    regime = choose_regime(
        prevalence, PREVALENCE_GATE, args.margin_factor)
    if regime == "conservative-veto":
        mode, threshold = "intersection", VETO_THRESHOLD
    elif regime == "recovery-add":
        mode, threshold = "union", ADD_THRESHOLD
    else:
        mode, threshold = "mapmos", ADD_THRESHOLD

    print("=== frozen label-free regime decision ===", flush=True)
    print(f"frames={frames} raw_points={total} moving_predictions={moving}",
          flush=True)
    print(f"raw_prevalence={prevalence:.8f} gate={PREVALENCE_GATE:.8f} "
          f"margin_factor={args.margin_factor:.3f}", flush=True)
    print(f"regime={regime} selected_mode={mode} "
          f"sparse_threshold={threshold:.6f}", flush=True)

    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "experiments" / "mapmos_sparse_fusion_eval.py"),
        "--config", args.config,
        "--ckpt-a", args.ckpt,
        "--ckpt-b", args.ckpt,
        "--mapmos-pred-dir", args.mapmos_pred_dir,
        "--weight-a", "1.0",
        "--pool", args.pool,
        "--thresholds", str(threshold),
        "--num-workers", str(args.num_workers),
    ]
    environment = os.environ.copy()
    process = subprocess.Popen(
        command, cwd=root, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    output = "".join(lines)
    if return_code:
        raise SystemExit(return_code)

    print("\n=== FROZEN REGIME-GATED RESULT ===")
    print(selected_row(output, mode))
    print("Decision inputs are prediction-only; GT was metric-only.")


if __name__ == "__main__":
    main()
