"""Collate 4D samples into a MinkowskiEngine batch.

ME expects coordinates as (N, 1 + D) with the first column = batch index.
Here D = 4 (x, y, z, t), so batched coords are (N, 5): [b, x, y, z, t].
"""

from __future__ import annotations
import numpy as np
import torch


def me_collate(batch):
    coords_list, feats_list, mot_list, sem_list, metas = [], [], [], [], []
    off_list, offm_list = [], []
    minst_list, instance_offset = [], 0
    has_pm = "ref_point_voxel" in batch[0]
    rpv_list, rpm_list = [], []
    voxel_offset = 0                       # rows already placed in the batch tensor
    for b, s in enumerate(batch):
        c = s["coords"]
        bcol = np.full((len(c), 1), b, dtype=np.int32)
        coords_list.append(np.concatenate([bcol, c], axis=1))  # (M, 5)
        feats_list.append(s["feats"])
        mot_list.append(s["motion"])
        sem_list.append(s["semantic"])
        off_list.append(s["offset"])
        offm_list.append(s["offset_mask"])
        minst = s.get("motion_instance")
        if minst is not None:
            minst = minst.astype(np.int64, copy=True)
            valid = minst >= 0
            if valid.any():
                _, local = np.unique(minst[valid], return_inverse=True)
                minst[valid] = local + instance_offset
                instance_offset += int(local.max()) + 1
            minst_list.append(minst)
        metas.append(s["meta"])
        if has_pm:
            # shift each sample's voxel indices into the concatenated row space
            rpv_list.append(s["ref_point_voxel"] + voxel_offset)
            rpm_list.append(s["ref_point_motion"])
        voxel_offset += len(c)
    coords = torch.from_numpy(np.concatenate(coords_list, 0)).int()
    feats = torch.from_numpy(np.concatenate(feats_list, 0)).float()
    motion = torch.from_numpy(np.concatenate(mot_list, 0)).long()
    semantic = torch.from_numpy(np.concatenate(sem_list, 0)).long()
    offset = torch.from_numpy(np.concatenate(off_list, 0)).float()
    offset_mask = torch.from_numpy(np.concatenate(offm_list, 0)).bool()
    out = {"coords": coords, "feats": feats,
           "motion": motion, "semantic": semantic,
           "offset": offset, "offset_mask": offset_mask, "meta": metas}
    if len(minst_list) == len(batch):
        out["motion_instance"] = torch.from_numpy(
            np.concatenate(minst_list, 0)).long()
    if has_pm:
        out["ref_point_voxel"] = torch.from_numpy(np.concatenate(rpv_list, 0)).long()
        out["ref_point_motion"] = torch.from_numpy(np.concatenate(rpm_list, 0)).long()
    return out
