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

    def iou(self):
        return self.inter / self.union.clamp(min=1)

    def miou(self):
        valid = self.union > 0
        return self.iou()[valid].mean().item() if valid.any() else 0.0

    def moving_iou(self):
        """IoU of class index 1 (the moving class) for MOS reporting."""
        return (self.inter[1] / self.union[1].clamp(min=1)).item()
