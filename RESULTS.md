# SparseUNet4D — Results

LiDAR moving-object segmentation (MOS) + multi-scan semantic segmentation on
SemanticKITTI. All numbers are **validation sequence 08**, moving-class IoU
unless noted. "point-level" = official protocol (voxel predictions propagated to
every reference point); "voxel" = deduplicated 4D voxels.

## Headline

| model | voxel IoU | point-level IoU | recall |
|-------|:---------:|:---------------:|:------:|
| baseline (starting point) | 0.625 | ~0.62 | 0.666 |
| **+ trajectory-consistent injection (inject2)** | **0.690** | **0.713** | **0.786** |

**0.62 → 0.713 point-level**, a +0.09 improvement whose largest single jump comes
from the injection method (below), not tuning.

## How we got there (and what didn't work)

The path was driven by a **diagnosis**, not trial-and-error:

**Diagnosis (`eval_fn_analysis.py`).** Of the moving voxels the baseline missed,
**55% are whole "silent" objects** — instances with near-zero residual signal,
not low-confidence detections. The threshold sweep (`eval_threshold_sweep.py`)
is flat (best 0.625 @ th 0.2 vs 0.623 @ 0.5): the model is well-calibrated, so
the misses are an **information limit**, not a calibration one. Instance-oracle
recall ceiling: 0.816 (IoU ~0.76).

**What moved the number:**

| step | voxel IoU | note |
|------|:---------:|------|
| baseline (v2), honest re-eval | 0.625 | argmax; threshold tuning worth ~0 |
| + strided temporal window `[1,2,4,8]` | 0.635 | widen receptive field to slow/near movers |
| + trajectory-consistent injection | **0.690** | attacks the silent-mover wall |

**What didn't (documented negatives — they sharpen the contribution):**

- **Geometric augmentation (rot/flip/scale):** −0.005 on val-08. Train and val
  are same-domain (same sensor/city), so augmentation only regularizes.
- **Test-time augmentation (D4):** −1.6 IoU on a non-augmented model (rotated
  views are out-of-distribution); +0.1 on the augmented model — confirms the
  mechanism but not worth the cost.
- **Imbalance-stacking:** `moving_class_weight 5 + dice + injection ×tripled
  density` collapses precision (first injection run: P 0.42). Rebalancing to
  `weight 2.5 / inject_prob 0.4 / max 2` recovers it → 0.690. (These
  compensations do not stack; each shifts the decision boundary toward moving.)
- **Two-model ensemble:** helps only when members are near-equal (temporal+aug
  → 0.666 point-level); adding a weaker correlated model to the strong inject2
  hurts (0.709 < 0.713).

## Contribution 1 — Trajectory-consistent 4D mover injection

`build_instance_bank.py` + `SemanticKITTI4D` (`inject_*`), `configs/residual_inject.yaml`.

Unlike single-scan instance pasting (e.g. InsMOS), each bank instance carries
its **real multi-frame trajectory**: points at the reference frame *and* each
strided past offset, GT-registered into the reference sensor frame. At training
time an instance is pasted with **one rigid yaw+translation applied to every
frame of its trajectory**, so its ego-compensated displacement — exactly what
the spherical-residual channels measure — is preserved. Injection runs *before*
residual computation, so synthesized movers get genuine motion signatures.

This raises training mover density ~3–5× (bank: 1106 instances, median
displacement 6.15 m over the widest offset, p10 0.68 m — includes slow movers),
directly targeting the diagnosed 55% silent-mover failure.

Result: recall **0.687 → 0.786** (+10 points), voxel IoU **0.635 → 0.690**,
point-level **0.713**. A single injection model beats the temporal+aug ensemble.

## Contribution 2 — Drift-consistency fine-tuning (ego-motion robustness)

`scripts/train_consistency.py`, `configs/consistency_ft.yaml`, swept by
`scripts/eval_robustness.py`.

Thesis premise, now demonstrated with data: **registration-dependent 4D MOS
models hallucinate motion under odometry drift.** The inject2 baseline collapses
from 0.690 clean to 0.024 under 2°/0.4 m per-step drift (a 96% collapse); moving
IoU craters *faster* than semantic mIoU, because motion detection leans entirely
on registration-based residuals.

Fix: fine-tune with **paired forward passes of the same sample** under clean vs
compounding-drift poses, adding a **mean-teacher consistency** term — the drift
prediction is pulled toward the *detached clean* prediction on the (provably
aligned) reference voxels. Registration only moves past frames, so the t=0 voxel
set is identical across the two passes; asymmetric + detached teacher removes the
degenerate constant solution symmetric KL allows.

### Robustness (moving IoU vs per-step drift), val-08

| drift (rot° / trans m) | inject2 base | consistency-FT | rel. gain |
|------------------------|:-----------:|:--------------:|:---------:|
| 0 / 0 (clean)          | 0.690 | 0.674 | −1.6 pts (cost) |
| 0.25 / 0.05            | 0.469 | **0.539** | +15% |
| 0.5 / 0.1              | 0.168 | **0.285** | **+70% (≈1.7×)** |
| 1 / 0.2                | 0.049 | 0.087 | +79% |
| 2 / 0.4                | 0.024 | 0.042 | +72% |

Consistency-FT is more robust at every drift level (crossover immediately after
0), ~1.7× at the realistic 0.5°/0.1 m level, for a **~1.6-point clean cost**.

**Scope (stated honestly):** at *catastrophic* drift (1–2°/step) both models are
effectively dead — no consistency training recovers motion once the residual cue
is destroyed. The gain is in the **realistic online-odometry regime
(0.25–0.5°/step)**, which is where real systems operate. Clean semantic mIoU also
dropped (0.42 → 0.38) as a side effect of the fine-tune; MOS was the target.

Ablation knobs in `consistency_ft.yaml`: `consistency_weight` /
`drift_sup_weight` isolate pure drift-augmentation vs pure invariance vs full
method.

## Reproduction

```bash
# 1. bank of moving instances with trajectories
python3 build_instance_bank.py --root <sequences> --seqs 0 1 2 3 4 5 6 7 9 10 \
    --offsets 1 2 4 8 --out mover_bank.npy
# 2. train with trajectory-consistent injection (headline model)
SU4D_BACKEND=me python scripts/train.py --config configs/residual_inject.yaml \
    --save-dir runs/residual_inject2
# 3. official point-level eval
SU4D_BACKEND=me python eval_mos_official.py --config configs/residual_inject.yaml \
    --ckpt runs/residual_inject2/best.pt --mos-yaml <semantic-kitti.yaml> \
    --point-level --threshold 0.5
# 4. drift-consistency fine-tune + robustness sweep
SU4D_BACKEND=me python scripts/train_consistency.py \
    --config configs/consistency_ft.yaml \
    --init runs/residual_inject2/best.pt --save-dir runs/consistency_ft
SU4D_BACKEND=me python scripts/eval_robustness.py \
    --config configs/residual_inject.yaml --ckpt runs/<ckpt>/best.pt --out <name>
```
