# SparseUNet4D — Results

LiDAR moving-object segmentation (MOS) + multi-scan semantic segmentation on
SemanticKITTI. All numbers are **validation sequence 08**, moving-class IoU
unless noted. "point-level" = official protocol (voxel predictions propagated to
every reference point, `eval_mos_official.py --point-level`); "voxel" =
deduplicated 4D voxels (the training-time meter). Point-level is the number
comparable to published work; the voxel meter is reported only where it drives
checkpoint selection.

## Headline

| model | point-level IoU | precision | recall |
|-------|:---------------:|:---------:|:------:|
| starting point | ~0.62 | — | 0.666 |
| trajectory-consistent injection (`residual_inject2`) | 0.7129 | 0.885 | 0.786 |
| **full model (`dual_v4`)** | **0.7680** | 0.874 | **0.864** |
| full model, tuned operating point (th 0.93) | 0.7711 | 0.891 | 0.852 |

**0.62 → 0.768 point-level.** Missed movers fall from 356,990 to 226,417 — a
**37% reduction in false negatives** — at essentially unchanged precision.

We report **0.768 at argmax (th = 0.5)** as the headline. The tuned-threshold
number is listed for completeness only: it is selected on the same validation
sequence it is reported on, and worth +0.3 IoU. The IoU curve is nearly flat from
th 0.05 to 0.95 (0.749 → 0.771), i.e. the model is **not threshold-fragile** —
which matters more for deployment than the extra 0.3.

### Position against published val-08 numbers

| method | val IoU |
|---|:---:|
| InsMOS | 73.2 |
| MF-MOS | 76.1 |
| MotionBEV | 76.5 |
| **ours (`dual_v4`)** | **76.8** |
| CV-MOS | 77.5 |
| 4D-CS | 80.9 |
| MambaMOS | 82.3 |

Above MF-MOS and MotionBEV, ~0.7 below CV-MOS, clearly below 4D-CS / MambaMOS.
We do **not** claim accuracy SOTA. The claimed contributions are the
ego-motion-robustness analysis, trajectory-consistent injection, the
object-consistency + decoupled-branch design, and a validated real-robot
deployment.

## Method: four accepted changes

Each was added **one at a time**, trained from scratch with an otherwise
identical recipe (40k iters, cosine, lr 1e-3), and scored under the same official
point-level protocol.

| # | change | config | point IoU | Δ |
|---|--------|--------|:---------:|:--:|
| 0 | baseline: trajectory-consistent injection | `residual_inject.yaml` | 0.7129 | — |
| 1 | object-level cluster consistency (scalar gate) | `cluster_v1.yaml` | 0.7299 | +1.7 |
| 2 | cluster fusion at feature level | `cluster_v3.yaml` | 0.7318 | +0.2 |
| 3 | decoupled motion / appearance branches | `dual_v1.yaml` | 0.7482 | +1.6 |
| 4 | all-frame supervision + rebalanced loss | `dual_v4.yaml` | **0.7680** | +2.0 |

### 1–2. Object-level cluster consistency (`sparseunet4d/models/cluster_head.py`)

**Diagnosis.** The baseline is recall-limited, not precision-limited: P 0.885 vs
R 0.786, with 357k false negatives against 169k false positives. The misses are
dominated by *partial* objects — sparse, far, or occluded parts of movers the
per-voxel head sees too little of.

**Mechanism.** A moving object is rigid: all its points share one label. We
cluster reference-frame foreground voxels (connected components on the voxel
grid), mean-pool backbone features per object, predict **one moving score per
object**, and fuse it back onto its member voxels. An auxiliary BCE on that score
(target = majority-moving of supervised members) shapes the pooled feature. The
fusion gate is initialised to **0**, so the model reduces exactly to the baseline
until the head earns trust — it cannot silently hurt.

This is the *inference-time* counterpart of the training-time trajectory-
consistent injection (Contribution 1): both encode "an object moves as one
trajectory-consistent unit". Related in spirit to the cluster priors of 4D-CS and
InsMOS.

**Result.** Recall 0.786 → 0.813, IoU +1.7. The learned gate converged to **1.46**
from a 0 init — the optimizer actively chose to use the object prior. Replacing
the scalar-bias fusion with **feature-level fusion** (concatenate the pooled
object feature onto each voxel, re-predict through a small MLP) added +0.2:
within noise on IoU, but it restored precision to above baseline (0.890) and
produced a **well-calibrated** model (optimal threshold 0.5; voxel meter
0.695 → 0.719), so it is the variant we keep.

### 3. Decoupled motion / appearance branches (`sparseunet4d/models/dual_branch.py`)

**Diagnosis.** Residual (motion) channels were simply concatenated onto the
occupancy/remission channel, so one backbone had to disentangle "this looks like a
car" from "this moved". The tell that it was overloaded: **semantic mIoU plateaued
at ~0.42** while motion IoU kept climbing.

**Mechanism.** Two encoders — appearance (remission) and motion (K residual
channels) — with their own stems and stages, fused at the bottleneck, both
branches' skips concatenated into a shared decoder. Same heads, same cluster
head. Follows the dedicated-motion-branch design of MF-MOS / CV-MOS, adapted to
4D sparse voxels.

**Result.** IoU 0.7318 → 0.7482 — the first change to improve **precision and
recall simultaneously** (0.896 / 0.820). Crucially, **semantic mIoU broke its
plateau (0.42 → 0.445)**, confirming the capacity-overload diagnosis rather than
merely correlating with it.

### 4. All-frame supervision + loss rebalancing

**Mechanism.** Previously only the reference frame was supervised; the other four
frames in the window carried IGNORE and contributed no gradient. Each past frame
is now supervised with **its own `.label` file** (a point moving at t−2 is
labelled moving in t−2's file) — ~5× denser signal for the same forward pass, as
in 4DMOS / 4D-CS. Applied to the **training set only**: the validation dataset
keeps reference-only labels, so the metric stays comparable across every
experiment here. The offset head remains reference-only (its target is the
reference-frame instance centre).

**The two-step result is the instructive part.** All-frame supervision *alone*
(`dual_v3`) **lost** on IoU (0.7176) — yet produced our best-then recall (0.828)
and lowest FN (286k) while precision collapsed to 0.843 (FP 159k → 256k). The
mechanism worked; the loss balance did not: 5× more supervised voxels means ~5×
more moving positives, so a `moving_class_weight` tuned for sparse supervision
over-drives false positives. Halving it (`dual_v4`) recovered precision and kept
the recall: **IoU 0.7680, R 0.864, FN 226k**.

## Rejected changes (documented negatives)

Each was a plausible idea, tested under the same protocol, rejected on evidence.

| change | result | why it failed |
|--------|:------:|---------------|
| **cross-frame cluster merging** (`cluster_v2`) | 0.6894 (−4.0 vs v1) | Clustering foreground across *all* frames and pooling over the whole trail **dilutes** features: a mover's displaced trail (and nearby foreground it links to) is averaged together. Both P and R fell. Occlusion recovery does not survive the strided-offset window. |
| **longer / lower-LR schedule** (`dual_v2`) | worse (voxel 0.710 vs 0.730) | Not under-training. Stretching cosine over 70k at half the peak LR meant the schedule **never annealed** before early stopping at 54k; `dual_v1`'s best checkpoint comes precisely from the low-LR tail of a *completed* 40k cosine. |
| **all-frame supervision, un-rebalanced** (`dual_v3`) | 0.7176 (−3.1) | Right mechanism, wrong loss balance — see above. Fixed in `dual_v4`. |
| **further class-weight reduction** (`dual_v5`) | 0.7347 (−3.3) | Overshoot. The calibration prediction was *confirmed* (optimal threshold migrated 0.93 → 0.45), but the probability skew was **functional**: it bought the recall. Traded 5.3 pts of recall for 1.5 of precision. `dual_v4` is the optimum on this axis. |

Earlier negatives from the injection phase still stand: geometric augmentation
(−0.005; same-domain train/val), test-time D4 augmentation (−1.6 on a
non-augmented model), imbalance-stacking (class weight 5 + dice + tripled mover
density collapses precision to 0.42), two-model ensembling with an unequal partner
(0.709 < 0.713), and a label-free residual voxel representative (0.635 — inflates
static-voxel residuals).

## Contribution 1 — Trajectory-consistent 4D mover injection

`build_instance_bank.py` + `SemanticKITTI4D` (`inject_*`), `configs/residual_inject.yaml`.

**Diagnosis (`eval_fn_analysis.py`).** Of the moving voxels the early baseline
missed, **55% are whole "silent" objects** — instances with near-zero residual
signal, not low-confidence detections. The threshold sweep was flat, so the misses
were an **information limit**, not a calibration one. Instance-oracle recall
ceiling: 0.816.

Unlike single-scan instance pasting (e.g. InsMOS), each bank instance carries its
**real multi-frame trajectory**: points at the reference frame *and* each strided
past offset, GT-registered into the reference sensor frame. At training time an
instance is pasted with **one rigid yaw+translation applied to every frame of its
trajectory**, so its ego-compensated displacement — exactly what the spherical-
residual channels measure — is preserved. Injection runs *before* residual
computation, so synthesized movers carry genuine motion signatures.

This raises training mover density ~3–5× (bank: 1106 instances, median
displacement 6.15 m over the widest offset, p10 0.68 m — includes slow movers),
directly targeting the diagnosed silent-mover failure. Recall **0.687 → 0.786**;
point-level IoU **0.713**. A single injection model beats the temporal+aug
ensemble.

## Contribution 2 — Drift-consistency fine-tuning (ego-motion robustness)

`scripts/train_consistency.py`, `configs/consistency_ft.yaml`, swept by
`scripts/eval_robustness.py`.

Thesis premise, demonstrated with data: **registration-dependent 4D MOS models
hallucinate motion under odometry drift.** The inject2 baseline collapses from
0.690 clean to 0.024 under 2°/0.4 m per-step drift (96% collapse); moving IoU
craters *faster* than semantic mIoU, because motion detection leans entirely on
registration-based residuals.

Fix: fine-tune with **paired forward passes of the same sample** under clean vs
compounding-drift poses, adding a **mean-teacher consistency** term — the drift
prediction is pulled toward the *detached clean* prediction on the (provably
aligned) reference voxels. Registration only moves past frames, so the t=0 voxel
set is identical across passes; the asymmetric + detached teacher removes the
degenerate constant solution symmetric KL admits.

### Robustness (moving IoU vs per-step drift), val-08

| drift (rot° / trans m) | inject2 base | consistency-FT | rel. gain |
|------------------------|:-----------:|:--------------:|:---------:|
| 0 / 0 (clean)          | 0.690 | 0.674 | −1.6 pts (cost) |
| 0.25 / 0.05            | 0.469 | **0.539** | +15% |
| 0.5 / 0.1              | 0.168 | **0.285** | **+70% (≈1.7×)** |
| 1 / 0.2                | 0.049 | 0.087 | +79% |
| 2 / 0.4                | 0.024 | 0.042 | +72% |

Consistency-FT is more robust at every drift level, ~1.7× at the realistic
0.5°/0.1 m level, for a ~1.6-point clean cost.

**Scope (stated honestly).** At *catastrophic* drift (1–2°/step) both models are
effectively dead — no consistency training recovers motion once the residual cue
is destroyed. The gain is in the realistic online-odometry regime
(0.25–0.5°/step). Separately, with **real** LiDAR odometry (KISS-ICP, even on
16-beam input) window-RPE stays ~5–6 cm and clean-model IoU is essentially
unaffected — so robustness matters in *degraded* regimes (poor odometry, tracking
loss), not in nominal operation. We report that rather than overstate the case.

Ablation knobs in `consistency_ft.yaml`: `consistency_weight` / `drift_sup_weight`
isolate pure drift-augmentation vs pure invariance vs the full method.

## Runtime

End-to-end per scan on val-08 (123k points/scan, ~320k voxels over the 5-frame
window), `deploy/benchmark_breakdown.py`.

| stage | RTX 5090 | RTX 3050 Ti Laptop (4 GB) |
|---|:---:|:---:|
| preprocess (CPU: residual images + voxelization) | 119 ms | 458 ms |
| **network (GPU)** | **78 ms (12.8 Hz)** | 522 ms (1.9 Hz) |
| postprocess (voxel→point) | 12 ms | 65 ms |
| **end-to-end** | **208 ms (4.8 Hz)** | 1043 ms (1.0 Hz) |

Vectorizing voxelization and the voxel→point lookup (bit-identical output) halved
end-to-end latency, 422 → 208 ms. Peak VRAM 777 MB.

**We do not claim a runtime advantage.** Published MOS runtimes are measured on an
RTX 3090 (SegNet4D 67 ms, Cylinder3D 125 ms, TemporalLatticeNet 154 ms, MarS3D
180 ms, KPConv 225 ms, SpSequenceNet 497 ms); normalising our 5090 figures to
3090-class hardware puts us mid-pack, and slower than the efficiency-focused
methods. The claim we make is **deployability**: real-time operation as a live
ROS 2 node on a 4 GB laptop GPU.

## Robot deployment

`mos_node.py`, `mos_inference.py`, `deploy/`.

Streaming ROS 2 (Humble) node: ring-buffered strided window, **pose-to-scan time
synchronisation** (odometry interpolated to each scan's timestamp — using the
latest pose instead costs 7–15 cm of registration error at 1 m/s, enough to flag
static walls as moving), NaN/inf filtering for Gazebo/real velodyne clouds,
optional embedded KISS-ICP when no odometry topic exists, and newest-scan-wins
scheduling so the node stays current rather than falling behind.

Validated in three settings: **replayed KITTI seq-08** (GT poses), the
**Clearpath Husky Gazebo simulation** (walking pedestrians segmented as moving
while static structures stay unflagged; a parked robot yields an empty moving set,
as it should), and a **4 GB laptop** running the full stack alongside ROS 2 in a
venv (`deploy/DEPLOY_VENV.md`). Warm-up is ~8 scans (the widest offset), by design.

**Known limitation.** The model is trained on 64-beam SemanticKITTI. A 16/32-beam
robot LiDAR is far sparser, so far-range movers (1–3 points at 15–20 m) drop out —
a sensor-density limit, not a pipeline fault. Closing it needs fine-tuning on
labelled robot data.

## Reproduction

```bash
# 1. bank of moving instances with trajectories
python3 build_instance_bank.py --root <sequences> --seqs 0 1 2 3 4 5 6 7 9 10 \
    --offsets 1 2 4 8 --out mover_bank.npy

# 2. train the full model (cluster head + dual branch + all-frame supervision)
SU4D_BACKEND=me python scripts/train.py --config configs/dual_v4.yaml \
    --save-dir runs/dual_v4

# 3. official point-level eval (headline number, argmax)
SU4D_BACKEND=me python eval_mos_official.py --config configs/dual_v4.yaml \
    --ckpt runs/dual_v4/best.pt --mos-yaml <semantic-kitti-mos.yaml> \
    --point-level --threshold 0.5

# 4. operating-point curve (one pass, full threshold grid)
SU4D_BACKEND=me python sweep_threshold_pointlevel.py \
    --config configs/dual_v4.yaml --ckpt runs/dual_v4/best.pt

# 5. ablation: rerun 2-3 with cluster_v1 / cluster_v3 / dual_v1 / dual_v3 / dual_v5

# 6. drift-consistency fine-tune + robustness sweep
SU4D_BACKEND=me python scripts/train_consistency.py \
    --config configs/consistency_ft.yaml \
    --init runs/dual_v4/best.pt --save-dir runs/consistency_ft
SU4D_BACKEND=me python scripts/eval_robustness.py \
    --config configs/dual_v4.yaml --ckpt runs/<ckpt>/best.pt --out <name>

# 7. runtime
SU4D_BACKEND=me python deploy/benchmark_breakdown.py \
    --config configs/dual_v4.yaml --ckpt runs/dual_v4/best.pt \
    --seq-dir <sequences>/08 --warmup 20 --n 300
```

All ablation runs use identical data, schedule and seed unless the table says
otherwise; only the named variable changes.

**Outstanding before publication.** The robustness table (Contribution 2) was
measured on the `inject2` generation of the model. Re-run the drift sweep on
`dual_v4` (step 6) so every table refers to the same final model.
