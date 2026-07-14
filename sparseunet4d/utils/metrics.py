"""Streaming IoU for motion (binary) and semantic (K-class). Ignores -1."""
from __future__ import annotations
import torch

IGNORE_INDEX = -1


class IoUMeter:
    def __init__(self, num_classes):
        self.k = num_classes
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.k)
        self.union = torch.zeros(self.k)
        self.pred_cnt = torch.zeros(self.k)   # TP + FP per class
        self.gt_cnt = torch.zeros(self.k)     # TP + FN per class

    def update(self, logits, labels):
        mask = labels != IGNORE_INDEX
        if mask.sum() == 0:
            return
        pred = logits[mask].argmax(1).cpu()
        gt = labels[mask].cpu()
        for c in range(self.k):
            p, g = pred == c, gt == c
            self.inter[c] += (p & g).sum()
            self.union[c] += (p | g).sum()
            self.pred_cnt[c] += p.sum()
            self.gt_cnt[c] += g.sum()

    def iou(self):
        return self.inter / self.union.clamp(min=1)

    def miou(self):
        valid = self.union > 0
        return self.iou()[valid].mean().item() if valid.any() else 0.0

    def moving_iou(self):
        """IoU of class index 1 (the moving class) for MOS reporting."""
        return (self.inter[1] / self.union[1].clamp(min=1)).item()
    
    def moving_pr(self):
        """(precision, recall) of class 1 for MOS reporting."""
        tp = self.inter[1]
        prec = (tp / self.pred_cnt[1].clamp(min=1)).item()
        rec = (tp / self.gt_cnt[1].clamp(min=1)).item()
        return prec, rec


class MovingThresholdMeter:
    """Streaming TP/FP/FN for the moving class over a grid of softmax
    thresholds, so validation can report (and select on) the IoU-optimal
    operating point instead of argmax@0.5. The model is heavily precision-
    skewed, so the best threshold is typically well below 0.5. Ignores -1."""

    def __init__(self, thresholds=(0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)):
        self.th = torch.tensor(list(thresholds), dtype=torch.float64)
        self.reset()

    def reset(self):
        k = len(self.th)
        self.tp = torch.zeros(k, dtype=torch.float64)
        self.fp = torch.zeros(k, dtype=torch.float64)
        self.fn = torch.zeros(k, dtype=torch.float64)

    def update(self, logits, labels):
        mask = labels != IGNORE_INDEX
        if mask.sum() == 0:
            return
        prob = torch.softmax(logits[mask].float(), dim=1)[:, 1].detach().cpu().double()
        gpos = (labels[mask] == 1).cpu()
        for k in range(len(self.th)):
            pr = prob >= self.th[k]
            self.tp[k] += float((pr & gpos).sum())
            self.fp[k] += float((pr & ~gpos).sum())
            self.fn[k] += float((~pr & gpos).sum())

    def iou_curve(self):
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def best(self):
        """Return the IoU-optimal threshold and its P/R, plus the argmax@0.5
        IoU for reference (threshold 0.5 is index 0)."""
        iou = self.iou_curve()
        k = int(torch.argmax(iou))
        prec = (self.tp[k] / (self.tp[k] + self.fp[k]).clamp(min=1)).item()
        rec = (self.tp[k] / (self.tp[k] + self.fn[k]).clamp(min=1)).item()
        return {"threshold": float(self.th[k]), "iou": float(iou[k]),
                "prec": prec, "rec": rec, "iou_argmax": float(iou[0])}
