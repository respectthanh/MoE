import torch, random, numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_hist_usage(probs=None, top1_idx=None, n_expert=4, title="usage"):
    if probs is not None:
        p_bar = probs.mean(dim=(0,1)).detach().cpu().numpy()
        plt.bar(range(len(p_bar)), p_bar); plt.title(title+" (soft p_bar)"); plt.show()
    if top1_idx is not None:
        counts = torch.bincount(top1_idx.reshape(-1), minlength=n_expert).float()
        p = (counts / counts.sum().clamp_min(1.0)).cpu().numpy()
        plt.bar(range(n_expert), p); plt.title(title+" (hard histogram)"); plt.show()
