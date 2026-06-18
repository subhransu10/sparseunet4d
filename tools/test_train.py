"""Smoke-test the training loop on synthetic 4D batches (mock backend)."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SU4D_BACKEND", "mock")
from sparseunet4d.models.backend import ST
from scripts.train import train

def fake_loader(n=4, B=2, npf=120, nf=3, ext=64, K=20):
    batches = []
    for _ in range(n):
        cl, fl, ml, sl = [], [], [], []
        for b in range(B):
            for t in range(nf):
                xyz = torch.randint(0, ext, (npf, 3))
                c = torch.cat([torch.full((npf,1),b), xyz, torch.full((npf,1),t)],1)
                cl.append(c); fl.append(torch.randn(npf,1))
                lab = torch.randint(0,2,(npf,)) if t==0 else torch.full((npf,),-1)
                sem = torch.randint(0,K,(npf,)) if t==0 else torch.full((npf,),-1)
                ml.append(lab); sl.append(sem)
        coords = torch.cat(cl).int()
        batches.append({"coords":coords,"feats":torch.cat(fl).float(),
                        "motion":torch.cat(ml).long(),"semantic":torch.cat(sl).long(),
                        "meta":[(0,0)]})
    return batches

cfg = {"dataset":{"num_semantic":20},
       "model":{"use_se":True,"use_ego_decouple":True},
       "train":{"lr":1e-3,"weight_decay":1e-4,"iters":20,"val_every":1},
       "loss":{"moving_class_weight":5.0,"dice_weight":1.0,"consistency_weight":1.0}}

print("Smoke-testing training loop...")
m = train(cfg, fake_loader(), val_loader=fake_loader(2),
          drift_loader=fake_loader(), device="cpu", max_iters=20, log_every=5)
print("Training loop OK.")
