# moe/core/runtime/combiner.py
from __future__ import annotations
from typing import List
import torch


@torch.no_grad()
def _add_weighted_(dst: torch.Tensor, src: torch.Tensor, idx: torch.Tensor, w: torch.Tensor):
    """
    Tiện ích: dst[sel] += w[sel].unsqueeze(-1) * src[idx[sel]]
    - Vector hóa theo expert để tránh vòng lặp theo token.
    """
    sel = idx >= 0
    if sel.any():
        dst[sel] += w[sel].unsqueeze(-1) * src[idx[sel]]


def simple_combine(
    expert_outputs: List[torch.Tensor],   # list[E], mỗi phần [C, H]
    topk_idx: torch.Tensor,               # [T, K] long, expert id đã chọn
    gating_weights: torch.Tensor,         # [T, K] float, softmax trên K logits
    dispatch_info: dict,                  # {'positions': [T,K] long, 'capacity': scalar}
    T: int,
) -> torch.Tensor:
    """
    Gom output từ experts về đúng thứ tự token ban đầu, nhân trọng số gating và sum theo K.

    Trả về:
      - out: Tensor [T, H]
    """
    device = expert_outputs[0].device if len(expert_outputs) > 0 else gating_weights.device
    H = expert_outputs[0].shape[-1] if len(expert_outputs) > 0 else 0
    K = topk_idx.shape[1]

    out = torch.zeros(T, H, device=device, dtype=expert_outputs[0].dtype)

    positions = dispatch_info["positions"]  # [T, K], -1 nếu bị drop

    # Thay vì lặp T*K cấp vi mô, ta lặp theo K và vector hóa theo expert.
    for j in range(K):
        e_ids = topk_idx[:, j]        # [T]
        pos_j = positions[:, j]       # [T], slot trong expert hoặc -1
        w_j = gating_weights[:, j]    # [T]

        # Duyệt theo từng expert để gather vector hóa
        # (vẫn là vòng lặp E, nhưng batch trên các token thuộc expert e)
        # Với E nhỏ (8–64) đây là lựa chọn tốt-nhanh-dễ.
        unique_e = torch.unique(e_ids)
        for e in unique_e.tolist():
            sel = (e_ids == e)
            if not sel.any():
                continue
            pos_e = pos_j[sel]              # [Ne]
            w_e = w_j[sel]                  # [Ne]
            # Chỉ lấy những slot hợp lệ
            valid = pos_e >= 0
            if not valid.any():
                continue
            pos_e = pos_e[valid]            # [M]
            w_e = w_e[valid]                # [M]

            # expert_outputs[e]: [C, H]
            src = expert_outputs[e]         # [C, H]
            # out[sel][valid] += w_e[:,None] * src[pos_e]
            # Để gán vào 'out' đúng vị trí toàn cục, ta lấy index toàn cục:
            idx_global = torch.nonzero(sel, as_tuple=False).squeeze(-1)[valid]  # [M]
            out[idx_global] += w_e.unsqueeze(-1) * src[pos_e]

    return out
