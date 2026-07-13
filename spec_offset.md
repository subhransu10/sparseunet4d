# Offset-head spec: learned instance grouping for MOS propagation

## Hypothesis (from measured data)
Oracle grouping = 0.769 IoU; best hand-crafted grouping = 0.643. The 12.6-pt
gap is grouping quality. A per-voxel offset to instance center, learned jointly,
should cluster better than 0.4m geometric linkage because it separates adjacent
instances (parked vs moving car) and connects fragmented ones.

## Milestone 1 (this build): offset head + center-space clustering in the
existing propagation harness. NOT full InsMOS. Success bar: beat 0.6430 in the
same F-sweep protocol.

## 1. Architecture delta (models/model.py) — ~5 lines
Add one linear head next to the existing two:
    self.offset_head = nn.Linear(widths[0], 3)     # (dx, dy, dz) in METERS
and in forward():
    "offset_pred": self.offset_head(feats),
No backbone change. Cost: 32*3 params. Everything else identical.

## 2. GT center generation (datasets/semantickitti.py) — reference frame only
In __getitem__, after labels exist and BEFORE voxelization concat:
  - take reference-frame points with mot==1 (moving)
  - cluster them with the SAME connected-components used in eval_fn_analysis
    (1.0m linkage on voxelized coords) -> instance ids
  - per instance: center = mean xyz (meters)
  - per point: gt_offset = center - xyz  (meters); non-moving & past-frame
    points get gt_offset = 0 and are MASKED from the offset loss
Return two new keys, carried through the same uniq indexing:
    "offset": (M,3) float32, "offset_mask": (M,) bool  (True only for
    reference-frame moving voxels)
Note: clustering ~5-40 instances/frame on <5k moving pts — negligible cost.
scipy import needed in the dataset module.

## 3. Loss (models/losses.py)
L1 on masked voxels, added to total:
    L_off = |offset_pred - offset_gt|_1 averaged over mask; skip if mask empty
    total += offset_weight * L_off        # config: loss.offset_weight: 1.0
Config key default 0.0 => exactly reproduces residual_v2 when off (clean A/B).
Expected magnitude: cars are ~4m long -> typical |offset| ~1m -> L_off starts
~1.0, comparable to CE; weight 1.0 is a sane start.

## 4. Collate (me_collate)
Concatenate "offset"/"offset_mask" across the batch like motion/semantic.

## 5. Eval: eval_instance_prop_v3
Copy of v2 with ONE change: cluster in shifted space
    shifted = xyz_m + offset_pred        (voxels vote for their center)
    components on floor(shifted / 0.3m)  (centers concentrate -> small linkage)
Same MOVABLE gate, same F sweep, same anchors (0.6430 / oracle 0.769).

## Incremental test plan (same discipline as residual build)
  T1 (CPU, no data): synthetic — two adjacent panels with different centers;
     verify gt_offset points at centers; verify offset loss decreases on
     overfit of 1 sample.
  T2 (mock_check-style): dataset returns offset keys, shapes/mask correct,
     offsets zero on past frames and statics, |offset| < ~6m sanity.
  T3 smoke: forward+loss+backward with offset_weight=1.0, grads finite.
  T4: 5k sanity run vs residual_v2's 5k curve — moving_iou must NOT degrade
     (offset head must be free w.r.t. MOS); offset L1 should fall below ~0.5.
  T5: full 40k, then eval_instance_prop_v3. Success: > 0.6430.

## Risks / expected failure modes
  - Offset noisy at low semantic quality -> centers smear -> clusters merge
    anyway. Mitigation: cluster shifted space at 0.3m; fall back to hybrid
    (geometric linkage AND center distance).
  - GT centers from connected components inherit the 1m-linkage flaws (two
    close movers = one GT instance). Acceptable for milestone 1; document it.