import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from moe.layers import SoftMoELayer
from moe.losses import lb_kl_uniform_from_probs
from moe.data import ToyAutoEncoderDataset
from moe.utils import set_seed, plot_hist_usage

def main():
    set_seed(0)
    device = "cpu"  # đổi 'cuda' nếu có GPU

    # HYPER
    BATCH = 16
    SEQ = 64
    D = 64
    H = 128
    E = 4
    LR = 1e-3
    LAMBDA_LB = 0.05
    EPOCHS = 10

    # data
    ds = ToyAutoEncoderDataset(n_seq=1024, T=SEQ, d_model=D, n_clusters=E, seed=0)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    # model
    moe = SoftMoELayer(d_model=D, d_hidden=H, n_expert=E).to(device)
    opt = torch.optim.AdamW(moe.parameters(), lr=LR)
    mse = nn.MSELoss()

    for ep in range(EPOCHS):
        moe.train()
        total_loss = total_task = total_lb = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)       # [B,T,D]
            y_pred, probs = moe(x)
            task_loss = mse(y_pred, y)
            lb_loss = lb_kl_uniform_from_probs(probs)
            loss = task_loss + LAMBDA_LB * lb_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_lb += lb_loss.item()

        print(f"[soft] epoch {ep+1}/{EPOCHS} loss={total_loss:.3f} task={total_task:.3f} lb={total_lb:.3f}")

    # xem phân phối sử dụng expert (kỳ vọng ~ đều)
    with torch.no_grad():
        x, _ = next(iter(dl))
        _, probs = moe(x.to(device))
        plot_hist_usage(probs=probs, n_expert=E, title="SoftMoE usage")

if __name__ == "__main__":
    main()
