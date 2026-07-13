"""Measure params + latency (ms/frame, FPS) for SparseUNet4D.

Matches the paper's runtime protocol: average forward-pass time over N iters
after a warmup. Runs on real ME with a real voxel count so numbers are honest.

Usage (5090):
  PYTHONPATH=~/MinkowskiEngine:~/sparseunet4d SU4D_BACKEND=me python3 benchmark.py \
      --config ~/sparseunet4d/configs/semantickitti_base.yaml \
      --ckpt ~/runs/full/best.pt --voxels 120000 --iters 300 --warmup 50
"""
import os, sys, time, argparse, yaml
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sparseunet4d"))
sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))
from sparseunet4d.models.backend import ST, backend
from sparseunet4d.models.model import SparseUNet4D


def count_params(m):
    n = sum(p.numel() for p in m.parameters())
    return n, n / 1e6


def make_input(nv, ext, nframes, device):
    per = nv // nframes
    cs, fs = [], []
    for t in range(nframes):
        xyz = torch.randint(0, ext, (per, 3))
        c = torch.cat([torch.zeros(per, 1, dtype=torch.long), xyz,
                       torch.full((per, 1), t, dtype=torch.long)], 1)
        cs.append(c); fs.append(torch.randn(per, 1))
    coords = torch.unique(torch.cat(cs), dim=0)
    feats = torch.randn(coords.shape[0], 1)
    if backend() == "me":
        import MinkowskiEngine as ME
        return ME.SparseTensor(feats.to(device), coordinates=coords.int().to(device))
    return ST(feats.to(device), coords.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--voxels", type=int, default=120000)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f); cfg.setdefault("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sem = cfg["dataset"].get("num_semantic", 20)

    model = SparseUNet4D(
        in_ch=4, num_semantic=num_sem,
        use_se=cfg["model"].get("use_se", True),
        use_ego_decouple=cfg["model"].get("use_ego_decouple", True)).to(device).eval()
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)

    n, nm = count_params(model)
    print(f"params: {n:,} ({nm:.2f} M)")

    ext = int(cfg["dataset"].get("point_range", 51.2) / cfg["dataset"]["voxel_size"])
    x = make_input(args.voxels, ext, cfg["dataset"]["n_frames"], device)

    def run():
        with torch.no_grad():
            if args.amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    model(x)
            else:
                model(x)

    for _ in range(args.warmup):
        run()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        run()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / args.iters * 1000
    print(f"latency: {ms:.2f} ms/frame  ({1000/ms:.2f} FPS)  "
          f"@ ~{args.voxels} voxels, {cfg['dataset']['n_frames']} frames"
          f"{' [AMP]' if args.amp else ''}")


if __name__ == "__main__":
    main()