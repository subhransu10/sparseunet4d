# ICRA 2027 ablation tables — frozen evidence

This document consolidates experiments already completed. Sequence 08 was a
one-shot internal audit and is no longer unseen; it must not be used for any
further selection. Final unbiased claims require the official hidden test set
or an external dataset.

## Table 1. Robust cluster-evidence ablation on development sequences 06/07

One shared threshold is selected by maximum worst-sequence point-level moving
IoU. All rows use the same label-free residual backbone checkpoint trained on
sequences 00–05, 09 and 10. Higher worst-sequence IoU is better.

| Pooling method | Shared threshold | Worst IoU | Macro IoU | Pooled IoU | Precision | Recall | Seq. 06 | Seq. 07 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point prediction (B0) | 0.09 | 0.7019 | 0.7030 | 0.7022 | 0.9283 | 0.7424 | 0.7040 | 0.7019 |
| Cluster mean | 0.15 | 0.7065 | 0.7089 | 0.7071 | 0.9302 | 0.7467 | 0.7113 | 0.7065 |
| **Cluster Top25 (B1)** | **0.20** | **0.7122** | **0.7143** | **0.7127** | **0.9292** | **0.7537** | **0.7164** | **0.7122** |
| Cluster Top10 | 0.50 | 0.7117 | 0.7237 | 0.7148 | 0.9352 | 0.7520 | 0.7357 | 0.7117 |
| Cluster maximum | 0.95 | 0.7101 | 0.7160 | 0.7116 | 0.9335 | 0.7496 | 0.7218 | 0.7101 |
| Learned cluster score | 0.20 | 0.7008 | 0.7038 | 0.7016 | 0.9277 | 0.7422 | 0.7069 | 0.7008 |

Top25 improves worst-sequence IoU by **+0.0103** over B0 and improves both
development sequences. It is the frozen B1 mechanism.

## Table 2. Mechanism controls for B1

This table separates point-threshold calibration from cluster completion.

| Method | Threshold | Worst IoU | Macro IoU | Pooled IoU | Precision | Recall | Seq. 06 | Seq. 07 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 point | 0.09 | 0.7019 | 0.7030 | 0.7022 | 0.9283 | 0.7424 | 0.7040 | 0.7019 | Baseline |
| Point-only control | 0.20 | 0.7012 | 0.7116 | 0.7038 | 0.9346 | 0.7403 | 0.7220 | 0.7012 | Calibration control |
| B1 Top25 | 0.20 | **0.7122** | **0.7143** | **0.7127** | 0.9292 | **0.7537** | 0.7164 | **0.7122** | Accept |
| Precision-controlled memory | 0.35 | 0.7122 | 0.7208 | 0.7144 | 0.9335 | 0.7527 | 0.7294 | 0.7122 | Reject: zero worst-sequence gain |

On sequence 06, moving from point threshold 0.09 to 0.20 removes 4,752 false
positives while losing 433 true positives. Top25 then adds 1,348 true positives
and 3,166 false positives. On sequence 07, Top25 adds 14,573 true positives for
3,600 false positives relative to the point-only control. Thus B1 provides a
real cross-sequence improvement, while the tested memory improves only the
non-bottleneck sequence.

## Table 3. Frozen development-to-audit generalization stress test

These results use the earlier single-development-sequence protocol. They are
reported as a negative generalization study, not as final test-set performance.

| Method | Dev. seq. 07 IoU | One-shot seq. 08 IoU | Change |
|---|---:|---:|---:|
| B0 point | 0.7175 | 0.7262 | +0.0087 |
| B1 cluster completion | 0.7211 | 0.6718 | -0.0493 |
| B2 causal memory | **0.7684** | 0.5819 | **-0.1865** |
| B4 reliability-filtered memory | 0.7670 | 0.5983 | -0.1687 |

This stress test motivated multi-sequence worst-case selection. It prevents the
paper from claiming a temporal gain that did not generalize.

## Table 4. Representation audit on hard moving clusters

AUROC compares hard moving clusters (frozen B1 Top25 score below 0.20) with all
predicted static clusters. Values are the minimum over sequences 06 and 07.
The predefined acceptance threshold was 0.70.

| Representation statistic | Legacy residual AUROC | 3×3 neighborhood residual AUROC |
|---|---:|---:|
| Absolute Top25 magnitude | 0.5053 | 0.5814 |
| Range-normalized Top25 magnitude | 0.4798 | 0.5955 |
| Coherent signed magnitude | 0.5347 | 0.5921 |
| Range-normalized coherent magnitude | 0.5078 | **0.6076** |
| Sign consistency | **0.5390** | 0.5280 |
| Temporal-offset support | 0.4075 | 0.5577 |
| Combined consistency | 0.5069 | 0.5841 |

Neither residual representation reaches the acceptance threshold. Adding a
residual MLP/head is therefore rejected.

## Table 5. Frozen ego-compensated object-trajectory audit

Reference clusters are associated one-to-one with same-class clusters in the
registered temporal slices using fixed physical gates. AUROC compares hard
moving clusters with predicted static clusters. The architecture decision was
predeclared on `trajectory_score`; the other rows are diagnostics only.

| Trajectory statistic | Seq. 06 AUROC | Seq. 07 AUROC | Worst AUROC |
|---|---:|---:|---:|
| Maximum displacement | 0.6144 | 0.5370 | 0.5370 |
| Furthest displacement | 0.6352 | 0.5337 | 0.5337 |
| Fitted speed | 0.6814 | 0.5335 | 0.5335 |
| Median speed | 0.6766 | 0.4990 | 0.4990 |
| Direction consistency | 0.5380 | 0.5029 | 0.5029 |
| Speed consistency | 0.5869 | 0.5230 | 0.5230 |
| Monotonicity | 0.5181 | 0.5277 | 0.5181 |
| Track support | 0.3540 | 0.5072 | 0.3540 |
| Displacement support | 0.5739 | 0.5367 | 0.5367 |
| **Trajectory score (predeclared)** | **0.6818** | **0.5476** | **0.5476** |

The predeclared statistic fails the 0.70 robust acceptance gate. Explicit
trajectory scoring is rejected without further tuning, and B1 Top25 is frozen
as the final development-selected method.

## Frozen paper claims supported today

1. A clean label-free multi-sequence protocol with one shared threshold and
   worst-sequence checkpoint selection.
2. Robust Top25 object completion improves worst-sequence moving IoU from
   0.7019 to 0.7122 on development sequences 06/07.
3. Single-sequence temporal calibration can fail catastrophically under a
   one-shot sequence shift; worst-sequence selection exposes this failure.
4. Per-ray residual magnitude, local angular matching and fixed
   ego-compensated trajectory statistics do not robustly separate the
   remaining hard movers.
5. B1 Top25 is the final development-selected method; residual, memory and
   trajectory extensions are rejected under predefined robust gates.
