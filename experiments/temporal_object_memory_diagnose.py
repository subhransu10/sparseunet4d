"""Mechanism diagnostics for frozen causal temporal object memory.

Compares three prediction paths in one deterministic pass:
  B0: raw point/voxel motion score,
  B1: current-frame cluster-max completion,
  B2: causal temporal object memory.

Ground truth is read only after all three predictions are produced.  Reports
official point-level totals, range and mover-class breakdowns, per-frame
instance detection buckets, and exact B0/B1 -> B2 error transitions.
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
from experiments.ensemble_eval import load_model, to_st
from experiments.temporal_object_memory_eval import (
    TemporalObjectMemory,
    build_proposals,
    cluster_complete,
)
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.datasets.label_map import split_label
from sparseunet4d.datasets.semantickitti import _read_label, _read_scan


MOV_NAME = {
    252: "car", 253: "bicyclist", 254: "person", 255: "motorcyclist",
    256: "on-rails", 257: "bus", 258: "truck", 259: "other-vehicle",
}
RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 52)]
MODES = ("B0-point", "B1-cluster", "B2-memory", "B3-hysteresis")


class Diagnostics:
    def __init__(self):
        n_modes, n_ranges = len(MODES), len(RANGE_BINS)
        self.tp = np.zeros(n_modes, np.int64)
        self.fp = np.zeros(n_modes, np.int64)
        self.fn = np.zeros(n_modes, np.int64)
        self.range_tp = np.zeros((n_modes, n_ranges), np.int64)
        self.range_fp = np.zeros_like(self.range_tp)
        self.range_fn = np.zeros_like(self.range_tp)
        self.class_tp = [defaultdict(int) for _ in MODES]
        self.class_fn = [defaultdict(int) for _ in MODES]
        # One entry per labelled moving instance in one frame.  This is not a
        # persistent GT track and is therefore called an instance-frame.
        self.instance_frames = []  # (n_points, [detected fraction per mode])
        self.moving_without_instance = 0
        self.transitions = {
            # recovered FN, lost TP, added FP, removed FP
            "B0->B2": np.zeros(4, np.int64),
            "B1->B2": np.zeros(4, np.int64),
            "B0->B3": np.zeros(4, np.int64),
            "B2->B3": np.zeros(4, np.int64),
        }

    def update(self, predictions, gt_motion, sem_raw, inst_raw, radius):
        valid = gt_motion != -1
        moving = (gt_motion == 1) & valid
        for mode, pred in enumerate(predictions):
            self.tp[mode] += int((pred & moving).sum())
            self.fp[mode] += int((pred & ~moving & valid).sum())
            self.fn[mode] += int((~pred & moving).sum())
            for index, (low, high) in enumerate(RANGE_BINS):
                selected = (radius >= low) & (radius < high) & valid
                self.range_tp[mode, index] += int(
                    (pred & moving & selected).sum())
                self.range_fp[mode, index] += int(
                    (pred & ~moving & selected).sum())
                self.range_fn[mode, index] += int(
                    (~pred & moving & selected).sum())
            for raw_id in np.unique(sem_raw[moving]):
                selected = moving & (sem_raw == raw_id)
                self.class_tp[mode][int(raw_id)] += int(
                    (pred & selected).sum())
                self.class_fn[mode][int(raw_id)] += int(
                    (~pred & selected).sum())

        b0, b1, b2, b3 = predictions
        comparisons = {
            "B0->B2": (b0, b2),
            "B1->B2": (b1, b2),
            "B0->B3": (b0, b3),
            "B2->B3": (b2, b3),
        }
        for label, (source, target) in comparisons.items():
            self.transitions[label] += [
                int((~source & target & moving).sum()),
                int((source & ~target & moving).sum()),
                int((~source & target & ~moving & valid).sum()),
                int((source & ~target & ~moving & valid).sum()),
            ]

        no_instance = moving & (inst_raw <= 0)
        self.moving_without_instance += int(no_instance.sum())
        for instance_id in np.unique(inst_raw[moving & (inst_raw > 0)]):
            selected = moving & (inst_raw == instance_id)
            n_points = int(selected.sum())
            if n_points >= 5:
                fractions = np.asarray([
                    float(pred[selected].mean()) for pred in predictions
                ])
                self.instance_frames.append((n_points, fractions))

    def print_report(self, thresholds, frames):
        print(f"\n==== overall ({frames} frames) ====")
        print(f"{'mode':>14} {'th':>10} {'IoU':>9} {'Prec':>9} "
              f"{'Rec':>9} {'TP':>10} {'FP':>10} {'FN':>10}")
        for index, mode in enumerate(MODES):
            tp, fp, fn = (int(self.tp[index]), int(self.fp[index]),
                          int(self.fn[index]))
            iou = tp / max(tp + fp + fn, 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            threshold_text = ("dual" if mode == "B3-hysteresis"
                              else f"{thresholds[index]:.5f}")
            print(f"{mode:>14} {threshold_text:>10} {iou:9.4f} "
                  f"{precision:9.4f} {recall:9.4f} {tp:10d} {fp:10d} "
                  f"{fn:10d}")

        print("\n==== B2 error transitions ====")
        for label, (recovered, lost_tp, added_fp,
                    removed_fp) in self.transitions.items():
            print(f"{label}: recovered moving points={int(recovered)}  "
                  f"lost moving points={int(lost_tp)}")
            print(f"{'':>8} added false positives={int(added_fp)}  "
                  f"removed false positives={int(removed_fp)}")
            print(f"{'':>8} net TP={int(recovered)-int(lost_tp):+d}  "
                  f"net FP={int(added_fp)-int(removed_fp):+d}")

        print("\n==== by RANGE ====")
        for mode_index, mode in enumerate(MODES):
            print(f"\n{mode}")
            print(f"{'bin(m)':>10} {'recall':>8} {'prec':>8} "
                  f"{'GTmov':>10} {'FN':>10} {'FP':>10}")
            for range_index, (low, high) in enumerate(RANGE_BINS):
                tp = self.range_tp[mode_index, range_index]
                fp = self.range_fp[mode_index, range_index]
                fn = self.range_fn[mode_index, range_index]
                print(f"{low:4d}-{high:<5d} {tp/max(tp+fn,1):8.3f} "
                      f"{tp/max(tp+fp,1):8.3f} {int(tp+fn):10d} "
                      f"{int(fn):10d} {int(fp):10d}")

        print("\n==== by MOVER CLASS ====")
        raw_ids = set()
        for mode_index in range(len(MODES)):
            raw_ids |= (self.class_tp[mode_index].keys()
                        | self.class_fn[mode_index].keys())
        raw_ids = sorted(
            raw_ids,
            key=lambda rid: -(self.class_tp[0][rid]
                              + self.class_fn[0][rid]))
        print(f"{'class':>14} {'GTpts':>10} "
              + " ".join(f"{mode:>12}" for mode in MODES))
        for raw_id in raw_ids:
            total = self.class_tp[0][raw_id] + self.class_fn[0][raw_id]
            recalls = []
            for mode_index in range(len(MODES)):
                tp = self.class_tp[mode_index][raw_id]
                fn = self.class_fn[mode_index][raw_id]
                recalls.append(tp / max(tp + fn, 1))
            print(f"{MOV_NAME.get(raw_id, raw_id):>14} {total:10d} "
                  + " ".join(f"{recall:12.3f}" for recall in recalls))

        print("\n==== by INSTANCE-FRAME ====")
        print("Each observation is one labelled object instance in one frame; "
              "these are not persistent GT tracks.")
        print(f"moving points without positive instance id: "
              f"{self.moving_without_instance}")
        if not self.instance_frames:
            return
        n_points = np.asarray([item[0] for item in self.instance_frames])
        fractions = np.stack([item[1] for item in self.instance_frames])
        buckets = [
            (0.0, 0.1, "missed entirely"),
            (0.1, 0.5, "mostly missed"),
            (0.5, 0.9, "partially found"),
            (0.9, 1.01, "fully found"),
        ]
        for mode_index, mode in enumerate(MODES):
            print(f"\n{mode}")
            print(f"{'bucket':>18} {'#obs':>7} {'%obs':>7} {'%mov pts':>10}")
            for low, high, label in buckets:
                selected = ((fractions[:, mode_index] >= low)
                            & (fractions[:, mode_index] < high))
                print(f"{label:>18} {int(selected.sum()):7d} "
                      f"{100*selected.mean():7.1f} "
                      f"{100*n_points[selected].sum()/max(n_points.sum(),1):10.1f}")

        b0_missed = fractions[:, 0] < 0.1
        for mode_index, label in ((2, "B2"), (3, "B3")):
            recovered = b0_missed & (fractions[:, mode_index] >= 0.1)
            print(f"\nB0 missed instance-frames recovered by {label} "
                  "(>=10% detected):")
            print(f"  observations: {int(recovered.sum())} / "
                  f"{int(b0_missed.sum())}")
            print(f"  moving points: {int(n_points[recovered].sum())} / "
                  f"{int(n_points[b0_missed].sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--point-threshold", type=float, default=0.005)
    parser.add_argument("--cluster-threshold", type=float, default=0.001)
    parser.add_argument("--memory-threshold", type=float, default=0.9998)
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--association-gate", type=float, default=1.5)
    parser.add_argument("--max-age", type=int, default=1)
    parser.add_argument("--velocity-alpha", type=float, default=0.5)
    parser.add_argument("--size-weight", type=float, default=0.5)
    parser.add_argument("--class-mismatch-penalty", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="zero evaluates every frame")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("model", {})
    d, p, model_cfg = cfg["dataset"], cfg["pose"], cfg["model"]
    if len(d["val_sequences"]) != 1:
        raise ValueError("diagnostic requires exactly one validation sequence")
    dataset = SemanticKITTI4D(
        d["root"], d["val_sequences"], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], p.get("mode", "gt"),
        p.get("rot_std_deg", 0.0), p.get("trans_std_m", 0.0),
        p.get("seed", 0), d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0),
        frame_offsets=d.get("frame_offsets"),
        feat_rep=d.get("feat_rep", "label"),
        residual_validity=d.get("residual_validity", False),
        residual_all_frames=d.get("residual_all_frames", False),
        return_point_map=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, d, model_cfg, device)
    memory = TemporalObjectMemory(
        alpha=args.alpha,
        association_gate=args.association_gate,
        max_age=args.max_age,
        velocity_alpha=args.velocity_alpha,
        size_weight=args.size_weight,
        class_mismatch_penalty=args.class_mismatch_penalty,
    )
    diagnostics = Diagnostics()
    thresholds = (args.point_threshold, args.cluster_threshold,
                  args.memory_threshold, float("nan"))
    frames = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            sequence, frame = (int(value) for value in batch["meta"][0])
            output = model(to_st(batch, device))
            voxel_score = torch.softmax(
                output["motion_logits"], 1)[:, 1].cpu().numpy()
            semantic_pred = output["semantic_logits"].argmax(1).cpu().numpy()
            cluster_ids = output["cluster_row_id"].cpu().numpy()
            coords5 = output["coords"].detach().cpu().numpy()
            proposals = build_proposals(
                voxel_score, cluster_ids, semantic_pred, coords5,
                dataset.pose_providers[sequence].pose(frame),
                float(d["voxel_size"]))
            current_scores = {
                proposal.cluster_id: proposal.score for proposal in proposals
            }
            memory_scores = memory.update(sequence, frame, proposals)

            point_voxel = batch["ref_point_voxel"].numpy()
            point_score = voxel_score[point_voxel]
            point_cluster = cluster_ids[point_voxel]
            cluster_score = cluster_complete(
                point_score, point_cluster, current_scores)
            temporal_score = cluster_complete(
                point_score, point_cluster, memory_scores)
            b0_prediction = point_score >= args.point_threshold
            b1_prediction = cluster_score >= args.cluster_threshold
            b2_prediction = temporal_score >= args.memory_threshold
            # Asymmetric hysteresis: weak point evidence is sufficient to
            # retain a detection, but new object-level completion requires the
            # much stronger temporal-memory threshold.
            b3_prediction = b0_prediction | b2_prediction
            predictions = [b0_prediction, b1_prediction, b2_prediction,
                           b3_prediction]

            sequence_dir = os.path.join(d["root"], f"{sequence:02d}")
            scan = _read_scan(os.path.join(
                sequence_dir, "velodyne", f"{frame:06d}.bin"))
            semantic_raw, instance_raw = split_label(_read_label(os.path.join(
                sequence_dir, "labels", f"{frame:06d}.label")))
            xyz = scan[:, :3]
            keep = np.all(np.abs(xyz) < d["point_range"], axis=1)
            gt_motion = batch["ref_point_motion"].numpy()
            if len(semantic_raw) != len(xyz) or int(keep.sum()) != len(gt_motion):
                raise RuntimeError(f"point alignment failed at {(sequence, frame)}")
            xyz = xyz[keep]
            diagnostics.update(
                predictions, gt_motion, semantic_raw[keep], instance_raw[keep],
                np.linalg.norm(xyz, axis=1))
            frames += 1
            if batch_index % 200 == 0:
                print(f"  frame {batch_index}/{len(loader)}", flush=True)
            if args.max_frames and frames >= args.max_frames:
                break

    print("\nPrediction path: frozen network + current clusters + causal odometry "
          "memory. GT is diagnostics-only.")
    print(f"B2 alpha={args.alpha:.2f} gate={args.association_gate:.2f} "
          f"max_age={args.max_age} velocity_alpha={args.velocity_alpha:.2f}")
    diagnostics.print_report(thresholds, frames)


if __name__ == "__main__":
    main()
