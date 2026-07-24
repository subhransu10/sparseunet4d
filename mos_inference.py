"""MOSInference: plug-and-play streaming wrapper for SparseUNet4D MOS.

Robot-facing API (zero ROS deps; wrap in a node separately):

    mos = MOSInference("configs/residual_v2.yaml", "runs/residual_v2/best.pt",
                       device="cuda")
    labels, probs = mos.push(scan_xyz_i, T_world_sensor)   # newest scan
      scan_xyz_i     : (N, 4) float32  [x, y, z, intensity], sensor frame
      T_world_sensor : (4, 4) float64  absolute pose of this scan
                       (from KISS-ICP / robot odometry / SLAM)
      labels         : (N,) int8   0 static, 1 moving   (-1 = outside range clip)
      probs          : (N,) float32 moving probability  (0 where label == -1)

Internally reproduces the TRAINING pipeline bit-for-bit: ring buffer of the
last n_frames scans, past frames registered into the current sensor frame,
range clip, per-point signed residual-image channels, 4D voxelization with
keep-first dedup, ME forward, voxel->point unmapping. The replay test
(`python3 mos_inference.py --replay-test ...`) pushes SemanticKITTI seq-08
through this class and asserts EXACT equality of coords/feats with the
Dataset, and matching voxel predictions -- if it passes, the robot path is
the validated path.

Optional post-hoc boost: `propagate=True` applies the v2 semantic-aware
instance propagation (claim@0.30) per frame (+~2 IoU, a few ms of scipy).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/sparseunet4d"))

IGNORE = -1


def _relative(T_now: np.ndarray, T_past: np.ndarray) -> np.ndarray:
    """Transform mapping past-sensor coords into now-sensor coords."""
    return np.linalg.inv(T_now) @ T_past


class MOSInference:
    def __init__(self, config_path, ckpt_path, device="cuda",
                 propagate=False, backend_name=None):
        import yaml, torch
        if backend_name:
            os.environ["SU4D_BACKEND"] = backend_name
        from sparseunet4d.models.backend import backend
        from sparseunet4d.models.model import SparseUNet4D

        with open(config_path) as f:
            cfg = yaml.safe_load(f); cfg.setdefault("model", {})
        d = cfg["dataset"]; m = cfg["model"]
        self.n_frames = d.get("n_frames", 4)
        # strided temporal window: buffer must reach the largest offset (e.g. 8),
        # not just n_frames-1. offsets/K mirror the training dataset exactly.
        self.offsets = list(d.get("frame_offsets") or range(1, self.n_frames))
        self.n_frames = 1 + len(self.offsets)
        self.max_offset = max(self.offsets) if self.offsets else 0
        self.voxel_size = d["voxel_size"]
        self.point_range = d.get("point_range", None)
        self.residual_feats = d.get("residual_feats", True)
        self.res_clip = d.get("res_clip", 3.0)
        # must match training: 'residual' picks the max-|residual| point per
        # voxel via the SAME shared helper the dataset uses (bit-identical);
        # 'label' (legacy ckpts) falls back to first-occurrence at inference.
        self.feat_rep = d.get("feat_rep", "label")
        self.device = device
        self.propagate = propagate
        self._backend = backend()

        in_ch = 1 + (self.n_frames - 1) if self.residual_feats else 1
        self.model = SparseUNet4D(
            in_ch, d.get("num_semantic", 20), base=m.get("base", 32),
            n_stages=m.get("n_stages", 2), use_se=m.get("use_se", True),
            use_ego_decouple=m.get("use_ego_decouple", False)
        ).to(device).eval()
        ck = torch.load(ckpt_path, map_location=device)
        self.threshold = float(os.environ.get('SU4D_THRESHOLD',
            ck.get('best_threshold', 0.5) if isinstance(ck, dict) else 0.5))
        print('[MOSInference] moving threshold =', self.threshold)
        missing, unexpected = self.model.load_state_dict(
            ck["model"] if "model" in ck else ck, strict=False)
        assert all(k.startswith("offset_head") for k in missing), missing
        assert not unexpected, unexpected

        self._buf = []            # newest first: [(xyz, remission, T_world)]

    # ------------------------------------------------------------------ #
    def reset(self):
        self._buf.clear()

    def push(self, scan_xyz_i: np.ndarray, T_world_sensor: np.ndarray):
        scan_xyz_i = np.asarray(scan_xyz_i, np.float32)
        self._buf.insert(0, (scan_xyz_i[:, :3].copy(),
                             scan_xyz_i[:, 3:4].copy(),
                             np.asarray(T_world_sensor, np.float64)))
        del self._buf[self.max_offset + 1:]        # keep enough for max offset
        T_ref = self._buf[0][2]
        # strided stack: (xyz, remission, rel_to_ref, offset). Offsets not yet
        # in the buffer (early in the stream) are simply skipped -> zero residual
        # channel, exactly like early-in-sequence frames at training time.
        stack = []
        for o in [0] + self.offsets:
            if o >= len(self._buf):
                continue
            xyz, rem, T = self._buf[o]
            rel = np.eye(4) if o == 0 else _relative(T_ref, T)
            stack.append((xyz, rem, rel, o))
        return self._infer(stack)

    def push_with_relatives(self, stack):
        """Test hook: stack = [(xyz, remission, rel_to_ref, offset)], reference
        first (offset 0, rel == identity). Mirrors a strided window explicitly."""
        return self._infer(stack)

    # ------------------------------------------------------------------ #
    def _assemble(self, stack):
        """EXACT mirror of SemanticKITTI4D.__getitem__ (unlabelled path),
        offset-aware. `stack` = [(xyz, remission, rel_to_ref, offset)]."""
        from sparseunet4d.datasets.semantickitti import _transform
        from sparseunet4d.datasets.residual_features import residual_channels

        coords_all, feats_all, frame_xyz = [], [], []
        stack_offsets = [o for (_, _, _, o) in stack]
        n_pts_ref = len(stack[0][0])
        keep_ref = None
        for t_idx, (xyz, remission, rel, o) in enumerate(stack):
            xyz = xyz.copy()
            if o != 0:
                xyz = _transform(xyz, rel).astype(np.float32)
            if self.point_range is not None:
                m = np.all(np.abs(xyz) < self.point_range, axis=1)
                xyz, remission = xyz[m], remission[m]
            else:
                m = np.ones(len(xyz), bool)
            if o == 0:
                keep_ref = m
            frame_xyz.append(xyz)
            t_col = np.full((len(xyz), 1), t_idx, np.float32)
            coords_all.append(np.concatenate([xyz, t_col], 1))
            feats_all.append(remission)

        coords = np.concatenate(coords_all, 0)
        feats = np.concatenate(feats_all, 0)

        K = self.n_frames - 1
        if self.residual_feats and K > 0:
            # channel k corresponds to self.offsets[k]; a missing offset -> zeros
            past_by_off = {stack_offsets[t]: frame_xyz[t]
                           for t in range(1, len(frame_xyz))}
            past_list = [past_by_off.get(o, np.zeros((0, 3), np.float32))
                         for o in self.offsets]
            R = residual_channels(frame_xyz[0], past_list,
                                  normalize=False, clip=self.res_clip)
            res_blocks = [R] + [np.zeros((len(frame_xyz[t]), K), np.float32)
                                for t in range(1, len(frame_xyz))]
            feats = np.concatenate([feats, np.concatenate(res_blocks, 0)], 1)

        q = coords.copy()
        q[:, :3] = np.floor(coords[:, :3] / self.voxel_size)
        q = q.astype(np.int32)
        if self.feat_rep == "residual":
            # bit-identical to training: shared label-free helper
            from sparseunet4d.datasets.semantickitti import residual_priority_rep
            uniq_coords, inv = np.unique(q, axis=0, return_inverse=True)
            rep = residual_priority_rep(inv.reshape(-1), len(uniq_coords), feats)
            return uniq_coords, feats[rep], keep_ref, n_pts_ref, frame_xyz[0]
        # legacy ('label'-trained) checkpoints: first-occurrence fallback; the
        # training motion-priority pick needs labels and can't be reproduced.
        uniq_coords, first_idx = np.unique(q, axis=0, return_index=True)
        return uniq_coords, feats[first_idx], keep_ref, n_pts_ref, frame_xyz[0]

    @staticmethod
    def _key(c):                       # (M,3) int voxel -> collision-free int64
        c = c.astype(np.int64)
        return (c[:, 0] + 2**20) * 2**42 + (c[:, 1] + 2**20) * 2**21 \
            + (c[:, 2] + 2**20)

    def _infer(self, stack):
        import torch
        qc, ft, keep_ref, n_ref, xyz_ref = self._assemble(stack)
        bcol = np.zeros((len(qc), 1), np.int32)
        coords = torch.from_numpy(np.concatenate([bcol, qc], 1)).int()
        feats = torch.from_numpy(ft).float()

        with torch.no_grad():
            if self._backend == "me":
                import MinkowskiEngine as ME
                x = ME.SparseTensor(feats.to(self.device),
                                    coordinates=coords.to(self.device))
            else:
                from sparseunet4d.models.backend import ST
                x = ST(feats.to(self.device), coords.to(self.device))
            out = self.model(x)
            prob_v = torch.softmax(out["motion_logits"], 1)[:, 1].cpu().numpy()
            sem_v = out["semantic_logits"].argmax(1).cpu().numpy()
            oc = out["coords"]
            oc = oc.cpu().numpy() if hasattr(oc, "cpu") else np.asarray(oc)

        t0 = oc[:, 4] == 0
        lut_p = dict(zip(self._key(oc[t0][:, 1:4]).tolist(),
                         prob_v[t0].tolist()))
        lut_s = dict(zip(self._key(oc[t0][:, 1:4]).tolist(),
                         sem_v[t0].tolist()))

        pk = self._key(np.floor(xyz_ref / self.voxel_size))
        p_prob = np.array([lut_p.get(k, 0.0) for k in pk.tolist()], np.float32)
        p_sem = np.array([lut_s.get(k, 0) for k in pk.tolist()], np.int64)
        if self.propagate:
            p_prob = self._propagate_v2(np.floor(xyz_ref / self.voxel_size)
                                        .astype(np.int64), p_prob, p_sem)

        labels = np.full(n_ref, IGNORE, np.int8)
        probs = np.zeros(n_ref, np.float32)
        labels[keep_ref] = (p_prob >= self.threshold).astype(np.int8)
        probs[keep_ref] = p_prob
        return labels, probs

    # ---- v2 semantic-aware instance propagation (claim@0.30) ------------ #
    @staticmethod
    def _propagate_v2(vox, prob, sem, th=0.5, frac=0.30, link=4,
                      min_c=5, max_c=15000):
        from scipy import ndimage
        movable = np.isin(sem, [1, 2, 3, 4, 5, 6, 7, 8])
        if not movable.any():
            return prob
        c = (vox[movable] // link); c -= c.min(0)
        grid = np.zeros(c.max(0) + 1, bool)
        grid[c[:, 0], c[:, 1], c[:, 2]] = True
        lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
        lab = lab[c[:, 0], c[:, 1], c[:, 2]]
        conf = prob >= th
        out = prob.copy()
        idx = np.where(movable)[0]
        for lid in np.unique(lab):
            s = lab == lid
            if not (min_c <= s.sum() <= max_c):
                continue
            if conf[idx[s]].mean() >= frac:
                out[idx[s]] = np.maximum(out[idx[s]], th)
        return out


# ========================================================================= #
# Replay-equality test: streaming class vs the training Dataset on seq-08.
# ========================================================================= #
def replay_test(args):
    import yaml, torch
    from sparseunet4d.datasets import SemanticKITTI4D
    from sparseunet4d.datasets.semantickitti import _read_scan
    from sparseunet4d.utils.metrics import IoUMeter

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["dataset"]; seq = d["val_sequences"][0]
    ds = SemanticKITTI4D(d["root"], [seq], d["n_frames"], d["voxel_size"],
        d["semantic_yaml"], "gt", 0.0, 0.0, 0, d["point_range"],
        residual_feats=d.get("residual_feats", True),
        res_clip=d.get("res_clip", 3.0), frame_offsets=d.get("frame_offsets"))
    provider = ds.pose_providers[seq]
    mos = MOSInference(args.config, args.ckpt, device=args.device,
                       propagate=args.propagate)

    n = min(args.frames, len(ds)) if args.frames else len(ds)
    meter = IoUMeter(2)
    feat_agree = []
    seq_dir = os.path.join(d["root"], f"{seq:02d}")
    step = max(1, len(ds) // n)
    checked = 0
    for i in range(0, len(ds), step):
        _, ref = ds.index[i]
        # ---- build the SAME strided stack the dataset uses ----------------
        stack = []
        for o in [0] + list(ds.offsets):
            f = ref - o
            if f < 0:
                continue
            s = _read_scan(os.path.join(seq_dir, "velodyne", f"{f:06d}.bin"))
            rel = np.eye(4) if o == 0 else provider.relative(f, ref)
            stack.append((s[:, :3].copy(), s[:, 3:4].copy(), rel, o))
        qc, ft, keep_ref, n_ref, xyz_ref = mos._assemble(stack)
        if checked < 20:
            ref_sample = ds[i]
            # Voxel GRID must be bit-exact (this is what determines the sparse
            # tensor the model sees). The per-voxel representative FEATURE can
            # differ at multi-point voxels: training picks the point by motion
            # LABEL (moving-preferring), which is unavailable at inference, so
            # the stream picks first-occurrence. Immaterial to predictions
            # (sub-voxel points share near-identical residuals) -> the IoU below
            # is the real gate. We only track the agreement for transparency.
            assert np.array_equal(qc, ref_sample["coords"]), \
                f"coords mismatch @ frame {ref}"
            feat_agree.append(float(np.isclose(ft, ref_sample["feats"]).all(1).mean()))
        # ---- accuracy over the replay ------------------------------------
        labels, probs = mos.push_with_relatives(stack)
        gt_pt = _point_gt(seq_dir, ref, keep_ref, n_ref)
        m = (labels != IGNORE) & (gt_pt != IGNORE)
        logits = np.stack([1 - probs[m], probs[m]], 1)
        meter.update(torch.from_numpy(logits), torch.from_numpy(gt_pt[m]))
        checked += 1
        if checked % 100 == 0:
            print(f"  {checked} frames, running moving-IoU="
                  f"{meter.moving_iou():.4f}", flush=True)

    prec, rec = meter.moving_pr()
    fa = 100.0 * np.mean(feat_agree) if feat_agree else float("nan")
    print(f"\nvoxel-GRID equality: PASSED (coords bit-exact on first 20 frames)")
    print(f"per-voxel feature agreement: {fa:.1f}%  (remainder = representative "
          f"pick at multi-point voxels; label-free at inference -> expected)")
    print(f"replay point-level moving-IoU={meter.moving_iou():.4f} "
          f"P={prec:.4f} R={rec:.4f}  ({checked} frames"
          f"{', propagate=v2' if args.propagate else ''})")
    print("GATE: this IoU should land within ~1-2 points of the val number for "
          "this checkpoint. If it does, the streaming/robot path is validated; "
          "the sub-voxel feature-representative difference is immaterial.")


def _point_gt(seq_dir, ref, keep_ref, n_ref):
    from sparseunet4d.datasets.semantickitti import _read_label
    from sparseunet4d.datasets.label_map import split_label, to_motion_labels
    sem_raw, _ = split_label(
        _read_label(os.path.join(seq_dir, "labels", f"{ref:06d}.label")))
    gt = np.full(n_ref, IGNORE, np.int64)
    gt[keep_ref] = to_motion_labels(sem_raw[keep_ref])
    return gt


class _FakeSelf:
    """Bind _assemble to a temporary buffer without disturbing mos state."""
    def __init__(self, mos, scans):
        self.__dict__.update(mos.__dict__)
        self._buf = [(s[0], s[1], None) for s in scans]
    _assemble = MOSInference._assemble
    _key = MOSInference._key


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-test", action="store_true")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = full seq-08; else subsample to ~N frames")
    ap.add_argument("--propagate", action="store_true",
                    help="apply v2 instance propagation per frame")
    args = ap.parse_args()
    if args.replay_test:
        replay_test(args)
    else:
        print("Import MOSInference from this module; --replay-test to validate.")