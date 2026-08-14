#!/usr/bin/env python3
"""One-shot evaluator for the frozen ICRA27 sequence-08 protocol.

This entry point intentionally exposes no calibration or method arguments.
All choices were frozen on development sequence 07 before evaluating
sequence 08. Only aggregate results are revealed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/icra27_backbone_sealed08_label_free.yaml"
CHECKPOINT = (
    ROOT / "runs/icra27_backbone_dev07_label_free_seed2701/best.pt"
)
FROZEN_RECORD = ROOT / "results/icra27_dev07_label_free_frozen.yaml"
DIAGNOSTIC = ROOT / "experiments/temporal_object_memory_diagnose.py"

EXPECTED_CHECKPOINT_SHA256 = (
    "103550a651ec968a4eae521836a75a47aa4663ced412573415d369e3df506a07"
)

POINT_THRESHOLD = 0.003
CLUSTER_THRESHOLD = 0.850
MEMORY_THRESHOLD = 0.002
ALPHA = 0.98
ASSOCIATION_GATE_M = 1.50
MAXIMUM_AGE_FRAMES = 1
RELIABILITY_MAXIMUM_SIZE_RATIO = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_protocol() -> None:
    for path in (CONFIG, CHECKPOINT, FROZEN_RECORD, DIAGNOSTIC):
        if not path.is_file():
            raise FileNotFoundError(path)

    with CONFIG.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset = cfg["dataset"]
    expected_train = [0, 1, 2, 3, 4, 5, 6, 9, 10]

    if dataset["train_sequences"] != expected_train:
        raise RuntimeError("frozen training split changed")
    if dataset["val_sequences"] != [8]:
        raise RuntimeError(
            "this evaluator is restricted to sealed sequence 08"
        )
    if dataset.get("feat_rep") != "residual":
        raise RuntimeError(
            "label-free residual feature representative required"
        )

    with FROZEN_RECORD.open(encoding="utf-8") as handle:
        record = yaml.safe_load(handle)

    method = record["selected_method"]
    expected = {
        "alpha": ALPHA,
        "association_gate_m": ASSOCIATION_GATE_M,
        "maximum_age_frames": MAXIMUM_AGE_FRAMES,
        "point_threshold": POINT_THRESHOLD,
        "cluster_threshold": CLUSTER_THRESHOLD,
        "memory_threshold": MEMORY_THRESHOLD,
    }

    for key, value in expected.items():
        if method[key] != value:
            raise RuntimeError(f"frozen method mismatch for {key}")

    if not record["protocol"].get("development_is_frozen", False):
        raise RuntimeError("development protocol is not marked frozen")

    actual_checkpoint_sha256 = sha256(CHECKPOINT)
    if actual_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint hash mismatch: " + actual_checkpoint_sha256
        )


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit(
            "This frozen evaluator accepts no command-line arguments"
        )

    verify_frozen_protocol()

    command = [
        sys.executable,
        str(DIAGNOSTIC),
        "--config", str(CONFIG),
        "--ckpt", str(CHECKPOINT),
        "--point-threshold", str(POINT_THRESHOLD),
        "--cluster-threshold", str(CLUSTER_THRESHOLD),
        "--memory-threshold", str(MEMORY_THRESHOLD),
        "--alpha", str(ALPHA),
        "--association-gate", str(ASSOCIATION_GATE_M),
        "--max-age", str(MAXIMUM_AGE_FRAMES),
        "--reliability-max-size-ratio",
        str(RELIABILITY_MAXIMUM_SIZE_RATIO),
    ]

    print("Frozen protocol verified.", flush=True)
    print(
        "Evaluating sealed SemanticKITTI sequence 08 exactly once.",
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    full_output = []
    aggregate_output = []
    capture_aggregate = False

    for line in process.stdout:
        full_output.append(line)
        stripped = line.strip()

        if stripped.startswith("frame "):
            print(line, end="", flush=True)

        if stripped.startswith("==== overall"):
            capture_aggregate = True
        elif (
            capture_aggregate
            and stripped.startswith("==== B2 error transitions")
        ):
            capture_aggregate = False

        if capture_aggregate:
            aggregate_output.append(line)

    return_code = process.wait()

    if return_code != 0:
        sys.stderr.write("".join(full_output))
        raise SystemExit(return_code)

    if not aggregate_output:
        sys.stderr.write("".join(full_output))
        raise RuntimeError("aggregate result block was not produced")

    print("\n=== FROZEN SEALED-SEQUENCE RESULT ===")
    print("".join(aggregate_output).rstrip())
    print("\nNo sequence-08 calibration or model selection is permitted.")


if __name__ == "__main__":
    main()
