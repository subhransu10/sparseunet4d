"""Losses for joint MOS + semantic, plus the drift-consistency term.

All losses respect IGNORE_INDEX (-1): only reference-frame voxels are
supervised. The consistency loss is the Phase-2 training piece that ties a
clean-pose forward pass to a drifted-pose forward pass on the same sample.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

IGNORE_INDEX = -1


def weighted_ce(logits, labels, class_weights=None):
    return F.cross_entropy(logits, labels, weight=class_weights,
                           ignore_index=IGNORE_INDEX)


def dice_moving(logits, labels, eps=1.0):
    """Soft Dice on the moving class (index 1), reference voxels only."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return logits.sum() * 0.0
    p = torch.softmax(logits[mask], dim=1)[:, 1]
    g = (labels[mask] == 1).float()
    num = 2 * (p * g).sum()
    den = p.sum() + g.sum() + eps
    return 1.0 - num / den


def consistency_loss(logits_clean, logits_drift):
    """Symmetric KL between motion distributions under clean vs drifted poses.

    Encourages the prediction to be invariant to ego-motion error. Assumes the
    two forward passes are over the SAME reference voxels in the same order.
    """
    p = torch.log_softmax(logits_clean, dim=1)
    q = torch.log_softmax(logits_drift, dim=1)
    pe, qe = p.exp(), q.exp()
    return 0.5 * (F.kl_div(q, pe, reduction="batchmean", log_target=False)
                  + F.kl_div(p, qe, reduction="batchmean", log_target=False))


def total_loss(out, motion_labels, semantic_labels, cfg,
               out_drift=None, offset_gt=None, offset_mask=None):
    w = None
    if cfg.get("moving_class_weight", 1.0) != 1.0:
        w = torch.tensor([1.0, cfg["moving_class_weight"]],
                         device=out["motion_logits"].device)
    l_mot = weighted_ce(out["motion_logits"], motion_labels, w)
    l_sem = weighted_ce(out["semantic_logits"], semantic_labels)
    if cfg.get("tversky", False):
        l_dice = tversky_moving(out["motion_logits"], motion_labels,
                                cfg.get("tversky_alpha", 0.3), cfg.get("tversky_beta", 0.7))
    else:
        l_dice = dice_moving(out["motion_logits"], motion_labels)
    loss = (cfg.get("motion_weight", 1.0) * l_mot
            + cfg.get("semantic_weight", 1.0) * l_sem
            + cfg.get("dice_weight", 1.0) * l_dice)
    parts = {"motion": l_mot.item(), "semantic": l_sem.item(),
             "dice": l_dice.item()}
    if out_drift is not None and cfg.get("consistency_weight", 0.0) > 0:
        l_con = consistency_loss(out["motion_logits"],
                                 out_drift["motion_logits"])
        loss = loss + cfg["consistency_weight"] * l_con
        parts["consistency"] = l_con.item()
    if (offset_gt is not None and cfg.get("offset_weight", 0.0) > 0
            and "offset_pred" in out):
        l_off = offset_l1(out["offset_pred"], offset_gt, offset_mask)
        loss = loss + cfg["offset_weight"] * l_off
        parts["offset"] = l_off.item()
    if cfg.get("cluster_weight", 0.0) > 0 and out.get("cluster_logits") is not None:
        l_clu = cluster_moving_loss(out.get("cluster_logits"),
                                    out.get("cluster_row_id"), motion_labels)
        if l_clu is not None:
            loss = loss + cfg["cluster_weight"] * l_clu
            parts["cluster"] = l_clu.item()
    return loss, parts


def cluster_moving_loss(cluster_logits, cluster_row_id, motion_labels):
    """BCE on the per-cluster moving score. Target = majority-moving of the
    cluster's supervised member voxels. Clusters with no supervised voxel are
    skipped. Aligns with the per-voxel motion labels (same row order)."""
    if cluster_logits is None or cluster_logits.numel() == 0:
        return None
    G = cluster_logits.shape[0]
    dev = cluster_logits.device
    valid = cluster_row_id >= 0
    ids = cluster_row_id[valid]
    lab = (motion_labels[valid] == 1).float()
    sup = (motion_labels[valid] != IGNORE_INDEX).float()
    pos = torch.zeros(G, device=dev).index_add_(0, ids, lab * sup)
    cnt = torch.zeros(G, device=dev).index_add_(0, ids, sup)
    has = cnt > 0
    if has.sum() == 0:
        return None
    target = (pos[has] / cnt[has].clamp(min=1.0) > 0.5).float()
    return F.binary_cross_entropy_with_logits(cluster_logits[has], target)

def tversky_moving(logits, labels, alpha=0.3, beta=0.7, eps=1.0):
    """Tversky on moving class (idx 1), ref voxels only. beta>alpha favours RECALL."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return logits.sum() * 0.0
    p = torch.softmax(logits[mask], dim=1)[:, 1]
    g = (labels[mask] == 1).float()
    tp = (p * g).sum(); fp = (p * (1 - g)).sum(); fn = ((1 - p) * g).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)

def offset_l1(offset_pred, offset_gt, offset_mask):
    """Masked L1 to instance center; mask = reference-frame moving voxels."""
    if offset_mask.sum() == 0:
        return offset_pred.sum() * 0.0
    return F.l1_loss(offset_pred[offset_mask], offset_gt[offset_mask])
