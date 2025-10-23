# moe/core/experts/mlp.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (không gamma/beta nếu không cần).
    Công thức: y = x / rms(x) * weight
    - Ổn định tốt cho mô hình decoder-only.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., H]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


class SwiGLUExpert(nn.Module):
    """
    Expert FFN kiểu SwiGLU (default cho MoE):
      x -> RMSNorm -> Linear(H, 2*Dff) split -> silu(a)*b -> Linear(Dff, H)
    Gợi ý: Dff ≈ 3.2 * H (giữa 2.67–3.5 đều ổn).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Khởi tạo nhẹ cho ổn định ban đầu
        nn.init.kaiming_uniform_(self.up.weight, a=math_silu_gain())
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        a, b = self.up(x).chunk(2, dim=-1)   # [T, d_ff], [T, d_ff]
        y = F.silu(a) * b
        y = self.drop(y)
        y = self.down(y)                      # [T, H]
        return y


def math_silu_gain() -> float:
    # gain gần đúng cho SiLU (không có trong nn.init.calculate_gain mặc định)
    # giá trị 1.0 là an toàn; 1.2 thường hợp lý với SwiGLU
    return 1.2


class LoRAExpert(nn.Module):
    """
    LoRA-per-Expert: dùng chung một FFN “base” ảo (cấu trúc), mỗi expert chỉ học
    2 adapter hạng thấp cho up/down để tiết kiệm tham số/VRAM.
    Triển khai tối giản: A (H->r) rồi B (r->out) và scale = alpha/r.

    Lưu ý:
    - Đây là expert “độc lập”, KHÔNG dùng chung weight với expert khác ở file này.
      Tiết kiệm đến từ rank r nhỏ, không phải weight sharing giữa experts.
    - Nếu bạn muốn “shared base + adapters per-expert”, ta sẽ tách base ra ngoài
      và truyền vào ở registry (có thể thêm sau).
    """
    def __init__(self, d_model: int, d_ff: int, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        # “Base” tuyến tính nhẹ (khởi tạo nhỏ), sau đó cộng adapter
        self.up_base   = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_base = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.up_base.weight)      # để phần LoRA chi phối ban đầu
        nn.init.zeros_(self.down_base.weight)

        # LoRA adapters
        self.A_up   = nn.Linear(d_model, r, bias=False)
        self.B_up   = nn.Linear(r, 2 * d_ff, bias=False)
        self.A_down = nn.Linear(d_ff, r, bias=False)
        self.B_down = nn.Linear(r, d_model, bias=False)
        self.scale = alpha / float(r)
        self.drop = nn.Dropout(dropout)

        # Khởi tạo LoRA
        nn.init.kaiming_uniform_(self.A_up.weight, a=math_silu_gain())
        nn.init.zeros_(self.B_up.weight)         # cấu hình LoRA chuẩn: B~0
        nn.init.kaiming_uniform_(self.A_down.weight, a=math_silu_gain())
        nn.init.zeros_(self.B_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        base_up = self.up_base(x)                               # [T, 2*Dff]
        lora_up = self.B_up(self.A_up(x)) * self.scale          # [T, 2*Dff]
        a, b = (base_up + lora_up).chunk(2, dim=-1)
        y = F.silu(a) * b
        y = self.drop(y)
        base_dn = self.down_base(y)                             # [T, H]
        lora_dn = self.B_down(self.A_down(y)) * self.scale      # [T, H]
        return base_dn + lora_dn
