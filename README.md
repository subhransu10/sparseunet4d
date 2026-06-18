# SparseUNet4D — Ego-Motion-Robust 4D LiDAR Segmentation

A 4D sparse-convolutional network for **joint** LiDAR moving-object segmentation
(MOS) and multi-scan semantic segmentation, built around one thesis:

> Existing 4D MOS/semantic methods are trained and evaluated with near-perfect
> SemanticKITTI poses. Their motion cue is "what fails to align after
> registration." Under realistic online-odometry drift, static structure stops
> aligning and these networks hallucinate motion. We characterise this failure
> for voxel-based 4D networks and design a model that separates ego-motion from
> object-motion instead of trusting registration.

The pose source is a **pluggable, perturbable** component (`datasets/poses.py`),
so the entire robustness study runs at inference on a fixed checkpoint — no
retraining needed for the headline figure.

## Status / roadmap
- [x] Pose handling: GT + compounding-drift providers (verified)
- [x] SemanticKITTI 4D dataset: pose-aware stacking, voxelization, joint labels
- [x] ME collate (4D batches)
- [ ] Model: 4D sparse U-Net backbone + dual motion/semantic heads
- [ ] Ego-motion decoupling module (the contribution)
- [ ] Training loop (weighted CE + Dice + optional consistency)
- [ ] `scripts/eval_robustness.py`: IoU-vs-drift sweep
- [ ] Baselines wiring (4DMOS, SegNet4D public checkpoints)

## Setup
```bash
# Use your existing 5090/cluster env if MinkowskiEngine is already built there.
pip install pyyaml numpy torch
# MinkowskiEngine: build from source against your CUDA/torch (it is finicky).
#   https://github.com/NVIDIA/MinkowskiEngine
# The 3050 Ti is for development + robustness analysis (inference); train on the 5090.
```

## Verify the pipeline (no dataset / no ME needed)
```bash
python tools/test_poses.py
```

## Notes that will save you debugging time
- KITTI poses are in the **camera** frame; we convert with `calib.txt` `Tr`.
  Getting this wrong silently misaligns everything — `test_poses.py` guards it.
- Only the reference frame (t == 0) is supervised; past frames are IGNORE_INDEX.
- Headline number must be a **clean same-protocol** SemanticKITTI-MOS test-set
  IoU — not a "+extended dataset" figure. The robustness curve is the novelty,
  not a higher pristine-pose number.
- Prior art to cite & beat: LiDAR-MOS (Chen 2021) has a single pose-jitter
  ablation; go beyond it with compounding drift, voxel-4D scope, a fix, and the
  joint task.
```
