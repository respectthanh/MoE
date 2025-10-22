# moe/core/routers/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# RoutingOutput: HỢP ĐỒNG DỮ LIỆU CHUẨN CHO MỌI ROUTER
# -----------------------------------------------------------------------------
@dataclass
class RoutingOutput:
    """
    Kết quả do Router trả về, được Orchestrator (MoELayer) sử dụng để:
    - dispatch (scatter) token tới experts,
    - combine (gather) đầu ra theo đúng thứ tự gốc,
    - cộng thêm auxiliary loss vào tổng loss,
    - ghi log diagnostics.

    Thuộc tính & shape:
      - dispatch_indices: LongTensor [T, k]
          Với mỗi token (T), chứa id của k expert được chọn theo thứ tự ưu tiên
          (cột 0 là expert quan trọng nhất). 0 <= id < num_experts.
      - gating_weights: FloatTensor [T, k]
          Softmax theo hàng trên k logits đã chọn. Tổng mỗi hàng ≈ 1.0.
          Dùng để weighted-sum các đầu ra chuyên gia khi combine.
      - aux_loss: Scalar tensor (0-dim) hoặc None
          Mất mát phụ trợ cho cân bằng tải, z-loss, v.v. Có thể là 0 nếu không dùng.
      - positions: LongTensor [T, k] hoặc None
          (Tuỳ chọn) vị trí slot của mỗi (token, expert) sau khi áp capacity trong
          dispatcher. Nếu router không sinh, dispatcher sẽ tự gán.
      - diagnostics: Dict[str, Tensor]
          Các chỉ số để quan sát/gỡ lỗi (vd: gate_logits_std, utilization, entropy,...).
    """
    dispatch_indices: torch.Tensor       # [T, k], long
    gating_weights: torch.Tensor         # [T, k], float
    aux_loss: Optional[torch.Tensor]     # scalar or None
    positions: Optional[torch.Tensor] = None  # [T, k], long
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# BaseRouter: LỚP TRỪU TƯỢNG (DESIGN-BY-CONTRACT)
# -----------------------------------------------------------------------------
class BaseRouter(nn.Module, ABC):
    """
    Mọi Router phải kế thừa lớp này và override phương thức `route`.

    Quy ước input:
      - x: FloatTensor [T, H] hoặc [B, S, H] (framework sẽ flatten về [T, H]).
        T = tổng số token trong batch (B*S), H = d_model.

    Quy ước output:
      - Trả về một `RoutingOutput` hợp lệ theo định nghĩa ở trên.

    Lưu ý:
      - Router KHÔNG thực thi capacity và KHÔNG giao tiếp all-to-all.
        Nó chỉ quyết định expert nào cho mỗi token và trọng số gating.
      - Router nên tính aux_loss (nếu có) và điền vào diagnostics các chỉ số có ích.
    """

    def __init__(self, *, num_experts: int, k: int = 1):
        super().__init__()
        if not (isinstance(num_experts, int) and num_experts > 0):
            raise ValueError("num_experts phải là số nguyên dương.")
        if not (isinstance(k, int) and 1 <= k <= num_experts):
            raise ValueError("k phải là số nguyên trong [1, num_experts].")
        self.num_experts = num_experts
        self.k = k

    @torch.no_grad()
    def _validate_output(
        self, out: RoutingOutput, T: int, device: torch.device
    ) -> None:
        """
        Kiểm tra tính hợp lệ tối thiểu của RoutingOutput để bắt bug sớm.
        - Kiểm tra dtype, shape, giá trị trong miền cho indices & weights.
        """
        # dispatch_indices
        if not isinstance(out.dispatch_indices, torch.Tensor):
            raise TypeError("dispatch_indices phải là torch.Tensor.")
        if out.dispatch_indices.dtype != torch.long:
            raise TypeError("dispatch_indices phải có dtype=torch.long.")
        if out.dispatch_indices.shape != (T, self.k):
            raise ValueError(f"dispatch_indices shape phải là [{T}, {self.k}].")
        if out.dispatch_indices.device != device:
            raise ValueError("dispatch_indices device không khớp input device.")
        if (out.dispatch_indices < 0).any() or (out.dispatch_indices >= self.num_experts).any():
            raise ValueError("dispatch_indices chứa expert id ngoài miền [0, num_experts).")

        # gating_weights
        if not isinstance(out.gating_weights, torch.Tensor):
            raise TypeError("gating_weights phải là torch.Tensor.")
        if out.gating_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("gating_weights phải là tensor float.")
        if out.gating_weights.shape != (T, self.k):
            raise ValueError(f"gating_weights shape phải là [{T}, {self.k}].")
        if out.gating_weights.device != device:
            raise ValueError("gating_weights device không khớp input device.")
        if torch.isnan(out.gating_weights).any() or torch.isinf(out.gating_weights).any():
            raise ValueError("gating_weights chứa NaN/Inf.")
        # kiểm tra tổng gần 1 theo hàng (khoan dung sai số số học)
        row_sum = out.gating_weights.sum(dim=-1)
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3, rtol=1e-3):
            raise ValueError("Mỗi hàng của gating_weights phải xấp xỉ tổng 1.0.")

        # aux_loss (nếu có)
        if out.aux_loss is not None:
            if out.aux_loss.dim() != 0:
                raise ValueError("aux_loss phải là scalar tensor (0-dim).")
            if out.aux_loss.device != device:
                raise ValueError("aux_loss device không khớp input device.")
            if torch.isnan(out.aux_loss) or torch.isinf(out.aux_loss):
                raise ValueError("aux_loss chứa NaN/Inf.")

        # positions (nếu có)
        if out.positions is not None:
            if out.positions.dtype != torch.long:
                raise TypeError("positions phải có dtype=torch.long.")
            if out.positions.shape != (T, self.k):
                raise ValueError(f"positions shape phải là [{T}, {self.k}].")
            if out.positions.device != device:
                raise ValueError("positions device không khớp input device.")
            if (out.positions < 0).any():
                raise ValueError("positions không được chứa giá trị âm.")

        # diagnostics: cho phép tự do nhưng không NaN/Inf
        for name, tensor in out.diagnostics.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"diagnostics['{name}'] phải là torch.Tensor.")
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"diagnostics['{name}'] chứa NaN/Inf.")

    @abstractmethod
    def route(self, x: torch.Tensor) -> RoutingOutput:
        """
        Tính quyết định định tuyến cho tensor đầu vào.

        Tham số:
          - x: FloatTensor [T, H] hoặc [B, S, H]
               Nếu input là [B, S, H], implementer có thể tự flatten về [T, H]
               HOẶC để cho MoELayer flatten trước khi gọi route(). Khuyến nghị:
               expect [T, H] để đơn giản.

        Trả về:
          - RoutingOutput (tuân thủ các yêu cầu ở trên).

        Lưu ý triển khai:
          - Nên dùng softmax trên K logits đã chọn để có gating_weights ổn định.
          - Có thể sinh thêm diagnostics như:
              * "gate_logits_std": std của logits trước softmax,
              * "utilization": phân phối tần suất chọn theo expert [E],
              * "entropy": entropy trung bình của phân phối gating, v.v.
        """
        ...

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        """
        Mặc định forward gọi route() và chạy validate (ở chế độ no_grad).
        Cho phép sử dụng Router như nn.Module thông thường.
        """
        # Chuẩn hoá shape về [T, H] nếu người dùng vô tình truyền [B,S,H]
        needs_flatten = (x.dim() == 3)
        if needs_flatten:
            B, S, H = x.shape
            x_ = x.reshape(B * S, H)
        elif x.dim() == 2:
            x_ = x
        else:
            raise ValueError("Input cho Router phải là [T,H] hoặc [B,S,H].")

        out = self.route(x_)
        # validate nhanh để bắt lỗi sớm trong dev; có thể tắt nếu cần tối ưu
        with torch.no_grad():
            self._validate_output(out, T=x_.size(0), device=x_.device)

        return out
