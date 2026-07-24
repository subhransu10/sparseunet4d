"""Benchmark per-scan runtime of SparseUNet4D MOS on this machine.

Reports END-TO-END latency (preprocess + network + postprocess) per scan on
real KITTI scans -- the runtime number MOS papers quote (4DMOS, MotionSeg3D,
MFMOS all report full per-scan time). Streams scans through the exact
deployment path (MOSInference.push), so the number reflects real inference,
not a synthetic forward pass. Run with SU4D_BACKEND=me.

Usage:
  SU4D_BACKEND=me python benchmark_runtime.py \
    --config configs/consistency_ft.yaml \
    --ckpt runs/consistency_ft/best.pt \
    --seq-dir /path/to/sequences/08 --warmup 20 --n 300
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from mos_inference import MOSInference
from sparseunet4d.datasets.poses import GTPoseProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-dir", required=True, help=".../sequences/08")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20,
                    help="scans to skip (fill the 5-frame buffer + GPU warm-up)")
    ap.add_argument("--n", type=int, default=300, help="scans to time")
    args = ap.parse_args()

    import torch
    cuda = args.device == "cuda" and torch.cuda.is_available()

    velo = os.path.join(args.seq_dir, "velodyne")
    files = sorted(f for f in os.listdir(velo) if f.endswith(".bin"))
    poses = GTPoseProvider(args.seq_dir).poses
    assert files, f"no .bin files in {velo}"

    mos = MOSInference(args.config, args.ckpt, device=args.device)
    if cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"scans available: {len(files)}  timing {args.n} after "
          f"{args.warmup} warm-up\n")

    times, npts = [], []
    total = min(args.warmup + args.n, len(files))
    for i in range(total):
        scan = np.fromfile(os.path.join(velo, files[i]),
                           np.float32).reshape(-1, 4)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mos.push(scan, poses[i])            # .cpu() inside forces GPU sync
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= args.warmup:
            times.append(dt); npts.append(len(scan))
        if i % 50 == 0:
            print(f"  scan {i}/{total}", flush=True)

    t = np.array(times)
    print("\n================= RUNTIME =================")
    if cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"scans timed:      {len(t)} (after {args.warmup} warm-up)")
    print(f"avg points/scan:  {int(np.mean(npts))}")
    print(f"latency (ms):     mean {t.mean():.1f}   median {np.median(t):.1f}"
          f"   p95 {np.percentile(t, 95):.1f}   "
          f"min {t.min():.1f}   max {t.max():.1f}")
    print(f"throughput (Hz):  mean {1000 / t.mean():.1f}   "
          f"median {1000 / np.median(t):.1f}")
    if cuda:
        print(f"peak VRAM (MB):   {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    print("==========================================")
    print("\nPaper-ready line, e.g.:")
    print(f'  "runs at {1000 / np.median(t):.1f} Hz '
          f'({np.median(t):.0f} ms/scan) on {torch.cuda.get_device_name(0) if cuda else "CPU"}"')


if __name__ == "__main__":
    main()
