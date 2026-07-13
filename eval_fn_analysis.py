"""FN characterization for MOS: WHY does the model miss 240k moving voxels?

One eval pass over val seq-08. For every GT-moving voxel, records:
  confidence (softmax moving prob), range, local density, residual magnitude.
Clusters GT-moving voxels into instances (connected components, 1m linkage)
and computes, per instance, the fraction of points the model finds.

Outputs:
  1. FN vs TP feature histograms (range / density / residual) -> what the
     misses look like.
  2. Instance-level breakdown: fully-found / partially-found / fully-missed.
  3. INSTANCE ORACLE upper bound: IoU/recall if every instance containing
     >=1 confident point were claimed entirely. This is the ceiling of any
     instance-propagation method (InsMOS-style). If oracle recall ~= current
     recall -> FNs are whole silent objects -> only a learned motion branch
     can help. If oracle recall >> current -> instance reasoning is the move.

Usage:
  cd ~/MinkowskiEngine
  SU4D_BACKEND=me PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d python3 \
    ~/sparseunet4d/eval_fn_analysis.py \
    --config ~/sparseunet4d/configs/residual_v2.yaml \
    --ckpt ~/sparseunet4d/runs/residual_v2/best.pt
"""
import os, sys, argparse, yaml
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.datasets import SemanticKITTI4D, me_collate
from sparseunet4d.models.backend import backend
from sparseunet4d.models.model import SparseUNet4D
from torch.utils.data import DataLoader
from scipy import ndimage

TH = 0.5          # decision threshold (matches argmax)
LINK_VOX = 10     # instance linkage: 10 voxels @0.1m = 1.0m connectivity


def cluster_instances(coords_xyz_vox):
    """Connected components of GT-moving voxels via coarse-grid labeling.
    coords in voxel units; coarsen by LINK_VOX so points within ~1m connect."""
    if len(coords_xyz_vox) == 0:
        return np.zeros(0, np.int64)
    c = (coords_xyz_vox // LINK_VOX).astype(np.int64)
    c -= c.min(0)
    shape = c.max(0) + 1
    grid = np.zeros(shape, dtype=bool)
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    return lab[c[:, 0], c[:, 1], c[:, 2]]


def local_density(pts, voxel=0.5):
    q = np.floor(pts / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv]


def hist_line(vals, edges, label):
    h, _ = np.histogram(vals, bins=edges)
    frac = h / max(h.sum(), 1)
    cells = " ".join(f"{f*100:4.0f}" for f in frac)
    print(f"  {label:11s} |{cells}|  (% per bin)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    d = cfg["dataset"]; p = cfg["pose"]; m = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTI4D(d["root"], d["val_sequences"], d["n_frames"],
        d["voxel_size"], d["semantic_yaml"], "gt", 0.0, 0.0, p["seed"],
        d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=me_collate, num_workers=4)

    n_frames = d.get("n_frames", 4)
    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1
    model = SparseUNet4D(in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
        n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
        use_ego_decouple=m.get("use_ego_decouple", False)).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    # accumulators over all GT-moving voxels
    conf_l, rng_l, den_l, res_l, inst_found_l = [], [], [], [], []
    n_static_fp = 0; n_static = 0
    inst_stats = []   # per instance: (n_pts, n_found, mean_range)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            coords = batch["coords"].to(dev); feats = batch["feats"].to(dev)
            if backend() == "me":
                import MinkowskiEngine as ME
                x = ME.SparseTensor(feats, coordinates=coords)
            else:
                from sparseunet4d.models.backend import ST
                x = ST(feats, coords)
            out = model(x)
            prob = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
            gt = batch["motion"].numpy()
            cnp = batch["coords"].cpu().numpy()
            fnp = batch["feats"].cpu().numpy()

            sup = gt != -1
            g = gt[sup]; pm = prob[sup]
            # coords: (batch, x, y, z, t) after me_collate -> xyz cols 1:4
            xyz_vox = cnp[sup][:, 1:4].astype(np.float64)
            xyz_m = xyz_vox * d["voxel_size"]
            res_mag = np.abs(fnp[sup][:, 1:]).max(1) if fnp.shape[1] > 1 else np.zeros(sup.sum())

            mov = g == 1
            n_static += int((~mov).sum()); n_static_fp += int((pm[~mov] >= TH).sum())
            if mov.sum() == 0:
                continue
            conf_l.append(pm[mov])
            rng_l.append(np.linalg.norm(xyz_m[mov], axis=1))
            den_l.append(local_density(xyz_m[mov]))
            res_l.append(res_mag[mov])

            inst = cluster_instances(xyz_vox[mov].astype(np.int64))
            found = pm[mov] >= TH
            for iid in np.unique(inst):
                sel = inst == iid
                inst_stats.append((int(sel.sum()), int(found[sel].sum()),
                                   float(np.linalg.norm(xyz_m[mov][sel], axis=1).mean())))
            # per-point: does this point's instance contain >=1 found point?
            found_per_inst = {iid: found[inst == iid].any() for iid in np.unique(inst)}
            inst_found_l.append(np.array([found_per_inst[i] for i in inst]))
            if bi % 400 == 0:
                print(f"  frame {bi}/{len(loader)}", flush=True)

    conf = np.concatenate(conf_l); rng = np.concatenate(rng_l)
    den = np.concatenate(den_l); res = np.concatenate(res_l)
    inst_found = np.concatenate(inst_found_l)
    is_tp = conf >= TH; is_fn = ~is_tp
    TPn, FNn = int(is_tp.sum()), int(is_fn.sum())
    print(f"\nGT-moving voxels: {TPn+FNn}  TP={TPn}  FN={FNn}  "
          f"recall={TPn/(TPn+FNn):.4f}")

    print("\n--- feature distributions, TP vs FN ---")
    e_r = [0, 10, 20, 30, 40, 51.2]
    print(f"  range bins (m): {e_r}")
    hist_line(rng[is_tp], e_r, "TP range"); hist_line(rng[is_fn], e_r, "FN range")
    e_d = [0, 5, 15, 40, 100, 1e9]
    print(f"  density bins (pts/0.5m vox): {e_d[:-1]}+")
    hist_line(den[is_tp], e_d, "TP density"); hist_line(den[is_fn], e_d, "FN density")
    e_s = [0, 0.05, 0.2, 0.5, 1.0, 3.01]
    print(f"  |residual| bins (m): {e_s}")
    hist_line(res[is_tp], e_s, "TP residual"); hist_line(res[is_fn], e_s, "FN residual")

    print("\n--- instance-level analysis ---")
    st = np.array(inst_stats, dtype=np.float64)  # n_pts, n_found, mean_range
    n_inst = len(st)
    frac_found = st[:, 1] / st[:, 0]
    full = (frac_found >= 0.9).sum(); zero = (frac_found == 0).sum()
    part = n_inst - full - zero
    print(f"  instances (>=1m linkage): {n_inst}")
    print(f"    fully found (>=90%): {full}  partial: {part}  FULLY MISSED: {zero}")
    zm = frac_found == 0
    if zm.any():
        print(f"    fully-missed: mean size {st[zm,0].mean():.0f} vox, "
              f"mean range {st[zm,2].mean():.1f} m "
              f"(vs found-inst mean range {st[~zm,2].mean():.1f} m)")
        print(f"    FN voxels inside fully-missed instances: "
              f"{int(st[zm,0].sum())} of {FNn} total FNs "
              f"({st[zm,0].sum()/max(FNn,1)*100:.0f}%)")

    print("\n--- INSTANCE ORACLE upper bound ---")
    # claim every voxel whose instance has >=1 TP point
    o_tp = int(inst_found.sum())
    o_fn = int((~inst_found).sum())
    FPn = n_static_fp   # FP unchanged by claiming GT-moving voxels
    o_rec = o_tp / max(o_tp + o_fn, 1)
    o_iou = o_tp / max(o_tp + FPn + o_fn, 1)
    print(f"  current : recall={TPn/(TPn+FNn):.4f}  (IoU 0.6235)")
    print(f"  oracle  : recall={o_rec:.4f}  IoU~{o_iou:.4f}   "
          f"(claim whole instance if any point found)")
    print(f"  -> instance propagation could recover at most "
          f"{(o_rec - TPn/(TPn+FNn))*100:.1f} recall points.")
    print("  If that number is large (>5-8 pts): build instance reasoning.")
    print("  If small: FNs are whole silent objects -> learned motion branch.")


if __name__ == "__main__":
    main()