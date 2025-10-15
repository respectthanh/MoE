import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from moe.layers import Top1MoELayer
from moe.losses import lb_kl_uniform_from_top1
from moe.data import ToyAutoEncoderDataset
from moe.utils import set_seed, plot_hist_usage

def main():
    set_seed(0)
    device = "cpu"

    # HYPER
    BATCH = 16
    SEQ = 64
    D = 64
    H = 128
    E = 4
    LR = 1e-3
    LAMBDA_LB = 0.05
    EPOCHS = 10
    CAP_FACTOR = 1.25

    ds = ToyAutoEncoderDataset(n_seq=1024, T=SEQ, d_model=D, n_clusters=E, seed=1)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    moe = Top1MoELayer(d_model=D, d_hidden=H, n_expert=E, capacity_factor=CAP_FACTOR).to(device)
    opt = torch.optim.AdamW(moe.parameters(), lr=LR)
    mse = nn.MSELoss()

    for ep in range(EPOCHS):
        moe.train()
        total_loss = total_task = total_lb = 0.0
        dropped_total = 0
        total_tokens = 0

        for x, y in dl:
            x, y = x.to(device), y.to(device)
            y_pred, probs, top1_idx = moe(x)
            task_loss = mse(y_pred, y)
            lb_loss = lb_kl_uniform_from_top1(top1_idx, n_expert=E)
            loss = task_loss + LAMBDA_LB * lb_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_lb += lb_loss.item()

            # ước lượng dropped-rate (gần đúng): token không được xử lý do capacity
            # ta coi token nào không có output khác 0 là bị drop (heuristic đơn giản)
            total_tokens += x.numel() // D
            dropped_total += (y_pred.abs().sum(dim=-1) == 0).sum().item()

        dropr = dropped_total / max(1, total_tokens)
        print(f"[top1] epoch {ep+1}/{EPOCHS} loss={total_loss:.3f} task={total_task:.3f} lb={total_lb:.3f} drop_rate≈{dropr:.4f}")

    with torch.no_grad():
        x, _ = next(iter(dl))
        _, probs, top1_idx = moe(x.to(device))
        plot_hist_usage(top1_idx=top1_idx, n_expert=E, title="Top1MoE usage")

if __name__ == "__main__":
    main()
