import torch

def lb_kl_uniform_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Load-balance cho Soft-MoE:
    probs: [B, T, E] -> lấy trung bình p_bar trên batch, áp KL(p_bar || uniform).
    Khuyến khích phân phối qua experts đồng đều.
    """
    p_bar = probs.mean(dim=(0,1))  # [E]
    E = p_bar.numel()
    uniform = torch.full_like(p_bar, 1.0/E)
    kl = (p_bar * (p_bar.clamp_min(1e-8)/uniform).log()).sum()
    return kl

def lb_kl_uniform_from_top1(top1_idx: torch.Tensor, n_expert: int) -> torch.Tensor:
    """
    Load-balance cho Hard Top-1:
    Đếm histogram assignment rồi áp KL(hist || uniform).
    """
    counts = torch.bincount(top1_idx.reshape(-1), minlength=n_expert).float()
    p = counts / counts.sum().clamp_min(1.0)
    uniform = torch.full_like(p, 1.0/n_expert)
    kl = (p * (p.clamp_min(1e-8)/uniform).log()).sum()
    return kl
