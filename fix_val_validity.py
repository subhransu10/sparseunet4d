"""Pass residual_validity to the VAL dataset too (train-only patch caused a
channel mismatch: model built for 1+2K, val_ds emitted 1+K).

Also fixes the two standalone eval scripts, which build their own datasets.
Idempotent. Run from repo root:  python fix_val_validity.py
"""
import os, shutil
R = os.path.expanduser("~/sparseunet4d")

TARGETS = [
    # (file, anchor that ends the SemanticKITTI4D(...) call)
    ("scripts/train.py",
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"))',
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"),\n'
     '        residual_validity=d.get("residual_validity", False))'),
    ("sweep_threshold_pointlevel.py",
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"))',
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"),\n'
     '        residual_validity=d.get("residual_validity", False))'),
    ("diagnose_errors.py",
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"))',
     '        frame_offsets=d.get("frame_offsets"), feat_rep=d.get("feat_rep", "label"),\n'
     '        residual_validity=d.get("residual_validity", False))'),
]

for path, old, new in TARGETS:
    p = os.path.join(R, path)
    if not os.path.exists(p):
        print(f"skip {path} (not found)"); continue
    s = open(p).read()
    if "residual_validity" in s and old not in s:
        print(f"skip {path} (already ok)"); continue
    if old not in s:
        print(f"WARN {path}: anchor not found - check manually"); continue
    shutil.copy(p, p + ".vvbak")
    open(p, "w").write(s.replace(old, new, 1))
    print("patched", path)

# also make in_ch consistent in the sweep script
p = os.path.join(R, "sweep_threshold_pointlevel.py")
if os.path.exists(p):
    s = open(p).read()
    old = '    in_ch = 1 + (n_frames - 1) if d.get("residual_feats", True) else 1'
    new = ('    _k = (n_frames - 1) * (2 if d.get("residual_validity", False) else 1)\n'
           '    in_ch = 1 + _k if d.get("residual_feats", True) else 1')
    if old in s:
        open(p, "w").write(s.replace(old, new, 1)); print("patched in_ch in sweep script")

# and in the diagnostic
p = os.path.join(R, "diagnose_errors.py")
if os.path.exists(p):
    s = open(p).read()
    old = '    in_ch = 1 + (d.get("n_frames", 4) - 1) if d.get("residual_feats", True) else 1'
    new = ('    _k = (d.get("n_frames", 4) - 1) * (2 if d.get("residual_validity", False) else 1)\n'
           '    in_ch = 1 + _k if d.get("residual_feats", True) else 1')
    if old in s:
        open(p, "w").write(s.replace(old, new, 1)); print("patched in_ch in diagnostic")

print("\ndone - val/eval datasets now match the model's channel count")
