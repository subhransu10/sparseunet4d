"""Break the per-scan runtime into preprocess / network / postprocess.

MOS papers quote the *network* inference time; the end-to-end number also
includes CPU-side preprocessing (residuals + voxelization) and postprocessing
(voxel->point lookup). This isolates each so you know which figure to report
and where the time actually goes. Run with SU4D_BACKEND=me.

Usage:
  SU4D_BACKEND=me python benchmark_breakdown.py \
    --config configs/consistency_ft.yaml --ckpt runs/consistency_ft/best.pt \
    --seq-dir /path/to/sequences/08 --warmup 20 --n 200
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
import mos_inference as mi
from mos_inference import MOSInference
from sparseunet4d.datasets.poses import GTPoseProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    import torch
    cuda = args.device == "cuda" and torch.cuda.is_available()
    velo = os.path.join(args.seq_dir, "velodyne")
    files = sorted(f for f in os.listdir(velo) if f.endswith(".bin"))
    poses = GTPoseProvider(args.seq_dir).poses
    mos = MOSInference(args.config, args.ckpt, device=args.device)
    import MinkowskiEngine as ME

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def build_stack(i):
        buf = []
        for j in range(i, max(-1, i - mos.max_offset - 1), -1):
            s = np.fromfile(os.path.join(velo, files[j]), np.float32).reshape(-1, 4)
            buf.append((s[:, :3].copy(), s[:, 3:4].copy(), poses[j]))
        T_ref = buf[0][2]
        stack = []
        for o in [0] + mos.offsets:
            if o < len(buf):
                xyz, rem, T = buf[o]
                rel = np.eye(4) if o == 0 else mi._relative(T_ref, T)
                stack.append((xyz, rem, rel, o))
        return stack

    tot, pre, net = [], [], []
    end = min(args.warmup + args.n, len(files))
    for i in range(mos.max_offset, end):
        scan = np.fromfile(os.path.join(velo, files[i]), np.float32).reshape(-1, 4)

        sync(); t0 = time.perf_counter()
        mos.push(scan, poses[i])                       # full end-to-end
        sync(); total = (time.perf_counter() - t0) * 1e3

        stack = build_stack(i)
        sync(); a = time.perf_counter()
        qc, ft, keep_ref, n_ref, xyz_ref = mos._assemble(stack)   # PREPROCESS
        sync(); pre_ms = (time.perf_counter() - a) * 1e3

        bcol = np.zeros((len(qc), 1), np.int32)
        coords = torch.from_numpy(np.concatenate([bcol, qc], 1)).int()
        feats = torch.from_numpy(ft).float()
        sync(); b = time.perf_counter()
        with torch.no_grad():
            x = ME.SparseTensor(feats.to(args.device),
                                coordinates=coords.to(args.device))
            out = mos.model(x)
            _ = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
        sync(); net_ms = (time.perf_counter() - b) * 1e3

        if i >= args.warmup:
            tot.append(total); pre.append(pre_ms); net.append(net_ms)
        if i % 50 == 0:
            print(f"  scan {i}/{end}", flush=True)

    tot, pre, net = map(np.array, (tot, pre, net))
    post = tot - pre - net                             # voxel->point LUT etc.
    def row(name, x):
        print(f"  {name:12s} mean {x.mean():7.1f}  median {np.median(x):7.1f}"
              f"  p95 {np.percentile(x,95):7.1f}   ms")
    print("\n============ RUNTIME BREAKDOWN (ms/scan) ============")
    if cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"scans timed: {len(tot)}   avg voxels/scan (5-frame): {len(qc)}")
    row("preprocess", pre)
    row("NETWORK", net)
    row("postprocess", post)
    row("END-TO-END", tot)
    print("=====================================================")
    print(f"\nNetwork-only throughput: {1000/np.median(net):.1f} Hz "
          f"(median {np.median(net):.0f} ms)")
    print(f"End-to-end throughput:   {1000/np.median(tot):.1f} Hz "
          f"(median {np.median(tot):.0f} ms)")


if __name__ == "__main__":
    main()
