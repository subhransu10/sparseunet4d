#!/usr/bin/env python3
"""Fixed-protocol diagnostics for B0 point versus B1 Top25 completion.

This script performs no model selection and no threshold sweep.  It evaluates
the already frozen shared development settings on each sequence separately:

* B0-point: point score >= 0.09
* point-control: point score >= 0.20
* B1-Top25: Top25 cluster completion >= 0.20

The point-control isolates the effect of changing threshold from the effect of
cluster completion.  Ground truth enters only the diagnostic accumulators.
Audited sequence 08 is forbidden.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from experiments.ensemble_cluster_eval import pooled_scores
from experiments.ensemble_eval import load_model, to_st
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import split_label
from sparseunet4d.datasets.semantickitti import _read_label, _read_scan


MODES = ("B0-point", "point-control", "B1-Top25")
MOVING_NAMES = {
    252: "car",
    253: "bicyclist",
    254: "person",
    255: "motorcyclist",
    256: "on-rails",
    257: "bus",
    258: "truck",
    259: "other-vehicle",
}
RANGE_BINS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 52))


def top25_complete(
    point_score: np.ndarray,
    point_cluster: np.ndarray,
    cluster_score: np.ndarray,
) -> np.ndarray:
    adjusted = point_score.copy()
    member = (point_cluster >= 0) & (point_cluster < len(cluster_score))
    adjusted[member] = np.maximum(
        adjusted[member], cluster_score[point_cluster[member]]
    )
    return adjusted


class Diagnostics:
    def __init__(self):
        mode_count = len(MODES)
        self.frames = 0
        self.tp = np.zeros(mode_count, np.int64)
        self.fp = np.zeros(mode_count, np.int64)
        self.fn = np.zeros(mode_count, np.int64)
        self.range_tp = np.zeros((mode_count, len(RANGE_BINS)), np.int64)
        self.range_fp = np.zeros_like(self.range_tp)
        self.range_fn = np.zeros_like(self.range_tp)
        self.class_tp = [defaultdict(int) for _ in MODES]
        self.class_fn = [defaultdict(int) for _ in MODES]
        self.instance_frames: list[tuple[int, np.ndarray]] = []
        self.moving_without_instance = 0
        self.transitions = {
            "B0->point-control": np.zeros(4, np.int64),
            "point-control->B1": np.zeros(4, np.int64),
            "B0->B1": np.zeros(4, np.int64),
        }

    def update(
        self,
        predictions: list[np.ndarray],
        motion_label: np.ndarray,
        semantic_raw: np.ndarray,
        instance_raw: np.ndarray,
        radius: np.ndarray,
    ) -> None:
        valid = motion_label != -1
        moving = (motion_label == 1) & valid
        for mode_index, prediction in enumerate(predictions):
            self.tp[mode_index] += int((prediction & moving).sum())
            self.fp[mode_index] += int(
                (prediction & ~moving & valid).sum()
            )
            self.fn[mode_index] += int((~prediction & moving).sum())
            for range_index, (low, high) in enumerate(RANGE_BINS):
                selected = (radius >= low) & (radius < high) & valid
                self.range_tp[mode_index, range_index] += int(
                    (prediction & moving & selected).sum()
                )
                self.range_fp[mode_index, range_index] += int(
                    (prediction & ~moving & selected).sum()
                )
                self.range_fn[mode_index, range_index] += int(
                    (~prediction & moving & selected).sum()
                )
            for raw_id in np.unique(semantic_raw[moving]):
                selected = moving & (semantic_raw == raw_id)
                self.class_tp[mode_index][int(raw_id)] += int(
                    (prediction & selected).sum()
                )
                self.class_fn[mode_index][int(raw_id)] += int(
                    (~prediction & selected).sum()
                )

        comparisons = {
            "B0->point-control": (predictions[0], predictions[1]),
            "point-control->B1": (predictions[1], predictions[2]),
            "B0->B1": (predictions[0], predictions[2]),
        }
        for name, (source, target) in comparisons.items():
            self.transitions[name] += np.asarray(
                [
                    int((~source & target & moving).sum()),
                    int((source & ~target & moving).sum()),
                    int((~source & target & ~moving & valid).sum()),
                    int((source & ~target & ~moving & valid).sum()),
                ],
                np.int64,
            )

        no_instance = moving & (instance_raw <= 0)
        self.moving_without_instance += int(no_instance.sum())
        for instance_id in np.unique(instance_raw[moving & (instance_raw > 0)]):
            selected = moving & (instance_raw == instance_id)
            point_count = int(selected.sum())
            if point_count >= 5:
                fractions = np.asarray(
                    [float(prediction[selected].mean())
                     for prediction in predictions],
                    np.float64,
                )
                self.instance_frames.append((point_count, fractions))
        self.frames += 1

    def metrics(self, mode_index: int) -> tuple[float, float, float]:
        tp = int(self.tp[mode_index])
        fp = int(self.fp[mode_index])
        fn = int(self.fn[mode_index])
        return (
            tp / max(tp + fp + fn, 1),
            tp / max(tp + fp, 1),
            tp / max(tp + fn, 1),
        )

    def print_report(
        self,
        sequence: int,
        point_threshold: float,
        cluster_threshold: float,
    ) -> None:
        thresholds = (point_threshold, cluster_threshold, cluster_threshold)
        print(f"\n{'=' * 72}\nSEQUENCE {sequence:02d} ({self.frames} frames)")
        print("=" * 72)
        print(
            f"{'mode':>15} {'th':>9} {'IoU':>9} {'Prec':>9} {'Rec':>9} "
            f"{'TP':>10} {'FP':>10} {'FN':>10}"
        )
        for index, mode in enumerate(MODES):
            iou, precision, recall = self.metrics(index)
            print(
                f"{mode:>15} {thresholds[index]:9.5f} {iou:9.4f} "
                f"{precision:9.4f} {recall:9.4f} "
                f"{int(self.tp[index]):10d} {int(self.fp[index]):10d} "
                f"{int(self.fn[index]):10d}"
            )

        print("\n---- exact error transitions ----")
        print(
            "Each row: recovered TP, lost TP, added FP, removed FP; "
            "net evidence is netTP-netFP."
        )
        for name, values in self.transitions.items():
            recovered, lost, added_fp, removed_fp = (int(x) for x in values)
            net_tp = recovered - lost
            net_fp = added_fp - removed_fp
            print(
                f"{name:>20}: recovered={recovered:8d} lost={lost:8d} "
                f"addedFP={added_fp:8d} removedFP={removed_fp:8d} "
                f"netTP={net_tp:+8d} netFP={net_fp:+8d} "
                f"evidence={net_tp-net_fp:+8d}"
            )

        print("\n---- by range ----")
        for mode_index, mode in enumerate(MODES):
            print(f"\n{mode}")
            print(
                f"{'bin(m)':>10} {'recall':>8} {'prec':>8} "
                f"{'GTmov':>10} {'FN':>10} {'FP':>10}"
            )
            for range_index, (low, high) in enumerate(RANGE_BINS):
                tp = self.range_tp[mode_index, range_index]
                fp = self.range_fp[mode_index, range_index]
                fn = self.range_fn[mode_index, range_index]
                print(
                    f"{low:4d}-{high:<5d} {tp/max(tp+fn, 1):8.3f} "
                    f"{tp/max(tp+fp, 1):8.3f} {int(tp+fn):10d} "
                    f"{int(fn):10d} {int(fp):10d}"
                )

        raw_ids: set[int] = set()
        for mode_index in range(len(MODES)):
            raw_ids |= (
                self.class_tp[mode_index].keys()
                | self.class_fn[mode_index].keys()
            )
        raw_ids = set(
            sorted(
                raw_ids,
                key=lambda raw_id: -(
                    self.class_tp[0][raw_id] + self.class_fn[0][raw_id]
                ),
            )
        )
        ordered_ids = sorted(
            raw_ids,
            key=lambda raw_id: -(
                self.class_tp[0][raw_id] + self.class_fn[0][raw_id]
            ),
        )
        print("\n---- by mover class (recall) ----")
        print(
            f"{'class':>14} {'GTpts':>10} "
            + " ".join(f"{mode:>15}" for mode in MODES)
        )
        for raw_id in ordered_ids:
            total = self.class_tp[0][raw_id] + self.class_fn[0][raw_id]
            recalls = []
            for mode_index in range(len(MODES)):
                tp = self.class_tp[mode_index][raw_id]
                fn = self.class_fn[mode_index][raw_id]
                recalls.append(tp / max(tp + fn, 1))
            print(
                f"{MOVING_NAMES.get(raw_id, str(raw_id)):>14} {total:10d} "
                + " ".join(f"{value:15.3f}" for value in recalls)
            )

        print("\n---- by instance-frame ----")
        print(
            "Each observation is one labelled instance in one frame, not a "
            "persistent GT track."
        )
        print(
            "moving points without positive instance id: "
            f"{self.moving_without_instance}"
        )
        if not self.instance_frames:
            return
        point_counts = np.asarray(
            [item[0] for item in self.instance_frames], np.int64
        )
        fractions = np.stack([item[1] for item in self.instance_frames])
        buckets = (
            (0.0, 0.1, "missed entirely"),
            (0.1, 0.5, "mostly missed"),
            (0.5, 0.9, "partially found"),
            (0.9, 1.01, "fully found"),
        )
        for mode_index, mode in enumerate(MODES):
            print(f"\n{mode}")
            print(f"{'bucket':>18} {'#obs':>7} {'%obs':>7} {'%mov pts':>10}")
            for low, high, name in buckets:
                selected = (
                    (fractions[:, mode_index] >= low)
                    & (fractions[:, mode_index] < high)
                )
                print(
                    f"{name:>18} {int(selected.sum()):7d} "
                    f"{100*selected.mean():7.1f} "
                    f"{100*point_counts[selected].sum()/max(point_counts.sum(),1):10.1f}"
                )

        b0_missed = fractions[:, 0] < 0.1
        for target_index, target_name in ((1, "point-control"), (2, "B1")):
            recovered = b0_missed & (fractions[:, target_index] >= 0.1)
            print(
                f"\nB0 missed instance-frames recovered by {target_name} "
                "(>=10% detected):"
            )
            print(
                f"  observations: {int(recovered.sum())} / "
                f"{int(b0_missed.sum())}"
            )
            print(
                f"  moving points: {int(point_counts[recovered].sum())} / "
                f"{int(point_counts[b0_missed].sum())}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--point-threshold", type=float, default=0.09)
    parser.add_argument("--cluster-threshold", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    dataset_cfg = cfg["dataset"]
    pose_cfg = cfg["pose"]
    model_cfg = cfg["model"]
    sequences = sorted(int(x) for x in dataset_cfg["val_sequences"])
    if len(sequences) < 2:
        raise ValueError("at least two development sequences are required")
    if 8 in sequences:
        raise ValueError("audited sequence 08 is forbidden for diagnostics")
    if not model_cfg.get("use_cluster", False):
        raise ValueError("config must enable the learned cluster head")

    dataset = SemanticKITTI4D(
        dataset_cfg["root"], sequences, dataset_cfg["n_frames"],
        dataset_cfg["voxel_size"], dataset_cfg["semantic_yaml"],
        "gt", 0.0, 0.0, pose_cfg["seed"], dataset_cfg["point_range"],
        residual_feats=dataset_cfg.get("residual_feats", True),
        res_clip=dataset_cfg.get("res_clip", 3.0),
        frame_offsets=dataset_cfg.get("frame_offsets"),
        feat_rep=dataset_cfg.get("feat_rep", "label"),
        residual_validity=dataset_cfg.get("residual_validity", False),
        residual_all_frames=dataset_cfg.get("residual_all_frames", False),
        return_point_map=True,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=me_collate,
        num_workers=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, dataset_cfg, model_cfg, device)
    diagnostics = {sequence: Diagnostics() for sequence in sequences}

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            sequence, frame = (int(x) for x in batch["meta"][0])
            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], dim=1
            )[:, 1].cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            cluster_score = pooled_scores(
                voxel_score, cluster_ids, "top25"
            )
            point_voxel = batch["ref_point_voxel"].numpy()
            motion_label = batch["ref_point_motion"].numpy()
            point_score = voxel_score[point_voxel]
            point_cluster = cluster_ids[point_voxel]
            completed_score = top25_complete(
                point_score, point_cluster, cluster_score
            )
            predictions = [
                point_score >= args.point_threshold,
                point_score >= args.cluster_threshold,
                completed_score >= args.cluster_threshold,
            ]

            sequence_dir = os.path.join(
                dataset_cfg["root"], f"{sequence:02d}"
            )
            scan = _read_scan(
                os.path.join(sequence_dir, "velodyne", f"{frame:06d}.bin")
            )
            xyz = scan[:, :3]
            keep = np.all(
                np.abs(xyz) < dataset_cfg["point_range"], axis=1
            )
            semantic_raw, instance_raw = split_label(
                _read_label(
                    os.path.join(
                        sequence_dir, "labels", f"{frame:06d}.label"
                    )
                )
            )
            if len(semantic_raw) != len(xyz) or int(keep.sum()) != len(
                motion_label
            ):
                raise RuntimeError(
                    f"reference-point alignment failed at {sequence:02d}/{frame:06d}"
                )
            xyz = xyz[keep]
            semantic_raw = semantic_raw[keep]
            instance_raw = instance_raw[keep]
            radius = np.linalg.norm(xyz, axis=1)
            diagnostics[sequence].update(
                predictions, motion_label, semantic_raw, instance_raw, radius
            )

            if frame_index % 200 == 0:
                print(f"  frame {frame_index}/{len(loader)}", flush=True)
            if args.max_frames and frame_index + 1 >= args.max_frames:
                break

    print(
        f"\nFrozen protocol: B0={args.point_threshold:.5f}, "
        f"B1 Top25={args.cluster_threshold:.5f}; no threshold search."
    )
    for sequence in sequences:
        diagnostics[sequence].print_report(
            sequence, args.point_threshold, args.cluster_threshold
        )

    print("\n" + "=" * 72)
    print("CROSS-SEQUENCE DECISION SUMMARY")
    print("=" * 72)
    print(f"{'sequence':>10} {'B0 IoU':>10} {'B1 IoU':>10} {'delta':>10}")
    for sequence in sequences:
        b0 = diagnostics[sequence].metrics(0)[0]
        b1 = diagnostics[sequence].metrics(2)[0]
        print(f"{sequence:10d} {b0:10.4f} {b1:10.4f} {b1-b0:+10.4f}")
    print(
        "Interpretation must be based on consistent mechanisms across both "
        "development sequences; audited sequence 08 remains forbidden."
    )


if __name__ == "__main__":
    main()
