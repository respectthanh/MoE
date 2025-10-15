import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Expert: một FFN đơn giản (Linear -> GELU -> Linear)
class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):  # x: [N, d_model]
        return self.net(x)

# ---- Router: tuyến theo token (token-level gating)
class Router(nn.Module):
    def __init__(self, d_model: int, n_expert: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_expert)

    def forward(self, x):  # x: [B, T, d_model]
        logits = self.gate(x)            # [B, T, E]
        probs = F.softmax(logits, dim=-1)  # phân phối mềm
        return logits, probs

class SoftMoELayer(nn.Module):
    """
    Soft routing: chạy tất cả experts rồi gộp theo trọng số probs.
    Dễ tối ưu vì mọi thứ phân biệt được (differentiable).
    """
    def __init__(self, d_model: int, d_hidden: int, n_expert: int):
        super().__init__()
        self.n_expert = n_expert
        self.router = Router(d_model, n_expert)
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(n_expert)])

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        logits, probs = self.router(x)  # [B,T,E]
        x_flat = x.reshape(B*T, D)

        # chạy mọi expert (đúng, nhưng tốn compute) rồi trộn theo probs
        expert_outs = []
        for e in self.experts:
            expert_outs.append(e(x_flat))    # [B*T, D] mỗi expert
        Y = torch.stack(expert_outs, dim=1)  # [B*T, E, D]
        probs_flat = probs.reshape(B*T, self.n_expert).unsqueeze(-1)  # [B*T,E,1]
        y = (probs_flat * Y).sum(dim=1)      # [B*T, D]
        return y.reshape(B, T, D), probs      # trả probs để tính LB loss
    
class Top1MoELayer(nn.Module):
    """
    Hard top-1: mỗi token chọn 1 expert (nhanh hơn).
    Dùng Straight-Through Estimator (STE) để backprop qua chọn-argmax.
    Có capacity_factor để tránh 1 expert bị 'quá tải'.
    """
    def __init__(self, d_model: int, d_hidden: int, n_expert: int, capacity_factor: float = 1.25):
        super().__init__()
        self.n_expert = n_expert
        self.router = Router(d_model, n_expert)
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(n_expert)])
        self.capacity_factor = capacity_factor

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        logits, probs = self.router(x)          # [B,T,E]
        top1_idx = torch.argmax(logits, dim=-1) # [B,T] expert chọn

        # STE: one-hot 'cứng' + gradient từ probs
        with torch.no_grad():
            one_hot = torch.nn.functional.one_hot(top1_idx, num_classes=self.n_expert).float()  # [B,T,E]
        hard_probs = one_hot - probs.detach() + probs  # trick: forward giống one-hot, backward giống probs

        # capacity: tối đa số token/expert mỗi bước
        total = B*T
        cap = int(self.capacity_factor * total / self.n_expert + 1)

        x_flat = x.reshape(B*T, D)
        y = torch.zeros_like(x_flat)
        top1_flat = top1_idx.reshape(-1)

        # Định tuyến từng expert (dễ hiểu; tối ưu sau)
        for e_id in range(self.n_expert):
            idx = (top1_flat == e_id).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            idx_use = idx[:cap]  # cắt nếu vượt capacity
            out = self.experts[e_id](x_flat[idx_use])
            y[idx_use] = out

        return y.reshape(B, T, D), probs, top1_idx
