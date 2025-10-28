# utils.py
import os, time, random
import numpy as np

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def default_paths(root=".", tag=None):
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    ckpt = os.path.join(root, "train", tag)
    logs = os.path.join(root, "logs", tag)
    ensure_dirs(ckpt, logs)
    return ckpt, logs
