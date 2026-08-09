"""Lovasz-softmax loss for the moving class.

WHY. We optimize CE + dice/Tversky, which are proxies for IoU. Lovasz-softmax is
a convex surrogate for the Jaccard index itself -- it optimizes the metric we are
scored on. It is standard in essentially every strong LiDAR segmentation method
(Cylinder3D, MinkUNet, MF-MOS, 4D-CS) and is the most conspicuous omission from
our recipe. Typically worth +1-2 IoU on this benchmark.

Enable with  loss.lovasz_weight: 1.0   (start at 1.0; dice_weight can stay)
Run once from repo root:  python apply_lovasz.py   (backs up losses.py.lvbak)
"""
import os, shutil
R = os.path.expanduser("~/sparseunet4d")
p = os.path.join(R, "sparseunet4d/models/losses.py")
s = open(p).read()
if "lovasz_softmax_moving" in s:
    raise SystemExit("already patched")
shutil.copy(p, p + ".lvbak")

LOVASZ = '''

def _lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss (Berman et al.)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_moving(logits, labels):
    """Lovasz-softmax on the MOVING class (index 1), reference voxels only.

    Convex surrogate of (1 - IoU) -- optimizes the evaluation metric directly
    rather than a proxy. Ignores IGNORE_INDEX voxels.
    """
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return logits.sum() * 0.0
    probs = torch.softmax(logits[mask], dim=1)[:, 1]
    fg = (labels[mask] == 1).float()
    if fg.sum() == 0:                      # no movers in this batch
        return logits.sum() * 0.0
    errors = (fg - probs).abs()
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    fg_sorted = fg[perm.detach()]
    return torch.dot(errors_sorted, _lovasz_grad(fg_sorted))
'''

# append the implementation
s = s.rstrip("\n") + "\n" + LOVASZ

# wire it into total_loss (before the cluster term / return)
old = """    if cfg.get("cluster_weight", 0.0) > 0 and out.get("cluster_logits") is not None:"""
new = """    if cfg.get("lovasz_weight", 0.0) > 0:
        l_lov = lovasz_softmax_moving(out["motion_logits"], motion_labels)
        loss = loss + cfg["lovasz_weight"] * l_lov
        parts["lovasz"] = l_lov.item()
    if cfg.get("cluster_weight", 0.0) > 0 and out.get("cluster_logits") is not None:"""
if old in s:
    s = s.replace(old, new, 1)
else:                                   # no cluster term present: hook before return
    old2 = "    return loss, parts\n\n\ndef cluster_moving_loss"
    new2 = ("""    if cfg.get("lovasz_weight", 0.0) > 0:
        l_lov = lovasz_softmax_moving(out["motion_logits"], motion_labels)
        loss = loss + cfg["lovasz_weight"] * l_lov
        parts["lovasz"] = l_lov.item()
    return loss, parts


def cluster_moving_loss""")
    assert old2 in s, "could not find a place to hook the lovasz term"
    s = s.replace(old2, new2, 1)

open(p, "w").write(s)
print("patched sparseunet4d/models/losses.py")
print("enable with  loss.lovasz_weight: 1.0")
