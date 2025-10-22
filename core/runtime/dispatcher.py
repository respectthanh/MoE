# moe/core/runtime/dispatcher.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math
import torch


@torch.no_grad()
def simple_dispatch(
    x: torch.Tensor,                 # [T, H]
    topk_idx: torch.Tensor,          # [T, K] long, expert id cho mỗi token
    *,
    capacity_factor: float = 1.25,
    drop_tokens: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], float]:
    """
    Gom token theo expert và áp dụng capacity.
    Trả về (dispatched_inputs, dispatch_info, dropped_frac):
      - dispatched_inputs: list[E] mỗi phần shape [C, H] (đã pack theo slot 0..C-1)
      - dispatch_info: {
            'positions': LongTensor [T, K]  (slot đã được gán; -1 nếu bị drop),
            'capacity':  LongTensor []      (scalar: C)
        }
      - dropped_frac: tỉ lệ (token,expert-slot) bị drop trên tổng T*K

    Ghi chú:
      - Bản single-device, dùng vòng lặp Python để dễ hiểu (đủ cho MVP).
      - Với bài toán lớn, ta sẽ thay bằng phiên bản vectorized / CUDA kernel / NCCL.
    """
    device = x.device
    T, H = x.shape
    E = int(torch.max(topk_idx).item()) + 1 if topk_idx.numel() > 0 else 0
    K = topk_idx.shape[1]

    if E <= 0:
        raise ValueError("Không tìm thấy expert id hợp lệ trong topk_idx.")

    # capacity per expert
    C = int(math.ceil(capacity_factor * T / E))
    if C <= 0:
        C = 1

    # Buffers per-expert
    dispatched_inputs: List[torch.Tensor] = [torch.zeros(C, H, device=device, dtype=x.dtype) for _ in range(E)]
    # per-expert slot counters
    counters = torch.zeros(E, dtype=torch.long, device=device)

    # positions[t, j] = slot được gán trong expert topk_idx[t,j], hoặc -1 nếu bị drop
    positions = torch.full((T, K), -1, dtype=torch.long, device=device)

    dropped = 0
    total_pairs = T * K

    # đơn giản: duyệt theo trật tự token → đảm bảo tính xác định
    # (khi tối ưu, ta sẽ vector hóa theo expert)
    for t in range(T):
        x_t = x[t]  # [H]
        for j in range(K):
            e = int(topk_idx[t, j].item())
            s = int(counters[e].item())
            if s < C:
                dispatched_inputs[e][s].copy_(x_t)
                positions[t, j] = s
                counters[e] = counters[e] + 1
            else:
                # expert đã đầy capacity
                if drop_tokens:
                    dropped += 1
                    # positions giữ -1
                else:
                    # Nếu không drop, bạn có thể chọn overwrite slot cuối (không khuyến nghị).
                    dropped += 1

    dropped_frac = float(dropped) / float(total_pairs) if total_pairs > 0 else 0.0

    dispatch_info = {
        "positions": positions,                 # [T, K]
        "capacity": torch.tensor(C, device=device, dtype=torch.long),  # scalar
    }
    return dispatched_inputs, dispatch_info, dropped_frac
