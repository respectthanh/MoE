# moe/core/routers/topk.py
from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRouter, RoutingOutput


class TopKRouter(BaseRouter):
    """
    Router Top-K kinh điển:
      - Tính logits = x @ W_g  (W_g là Linear learnable)
      - Chọn k logits lớn nhất cho mỗi token
      - Softmax trên k logits đã chọn -> gating_weights
      - Tính auxiliary load-balance loss (alpha * E * sum(P_i * f_i))
      - Trả về RoutingOutput cho Orchestrator/Dispatcher

    Tham số:
      d_model        : kích thước ẩn H của token
      num_experts    : số expert E
      k              : số expert được chọn cho mỗi token (1 hoặc 2 phổ biến)
      alpha          : hệ số cho load-balance auxiliary loss
      temperature    : hệ số chia logits trước softmax (mềm hóa; 1.0 là mặc định)
      zloss_beta     : nếu >0, thêm z-loss để ổn định logits (tránh quá lớn)

    Ghi chú:
      - route(x) kỳ vọng x shape [T, H]; nếu [B, S, H] đã được BaseRouter.forward xử lý.
      - Không thực thi capacity, không all-to-all; chỉ quyết định routing.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        k: int = 1,
        alpha: float = 1e-2,
        *,
        temperature: float = 1.0,
        zloss_beta: float = 0.0,
    ):
        super().__init__(num_experts=num_experts, k=k)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.zloss_beta = float(zloss_beta)

        # init nhẹ để logits đầu nhỏ & gần đều -> tránh "expert chết" sớm
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def _compute_aux_lb(self, topk_idx: torch.Tensor, gw: torch.Tensor, E: int) -> torch.Tensor:
        """
        Tính load-balancing auxiliary loss:
          L_aux = alpha * E * sum_i (P_i * f_i)
        Trong đó:
          f_i: fraction token được route tới expert i (theo Top-K chọn)
          P_i: tổng gating weight gán cho i, chia cho T
        """
        T, K = topk_idx.shape
        device = topk_idx.device

        # f_i: đếm tần suất expert được chọn (one_hot trên [T,E]) rồi mean theo T
        one_hot = torch.zeros(T, E, device=device, dtype=gw.dtype)
        one_hot.scatter_(1, topk_idx, 1.0)
        f = one_hot.mean(dim=0)  # [E]

        # P_i: tổng các trọng số gw góp cho expert i, sau đó /T
        p = torch.zeros(E, device=device, dtype=gw.dtype)
        p.index_add_(0, topk_idx.reshape(-1), gw.reshape(-1))
        p = p / float(T)

        aux = self.alpha * E * torch.sum(p * f)
        return aux

    def _compute_zloss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Z-loss cổ điển: mean(logsumexp(logits)^2)
        Mục tiêu: hạn chế logits quá lớn (ổn định số học/gradient)
        """
        if self.zloss_beta <= 0.0:
            return logits.new_tensor(0.0)
        # logsumexp theo trục expert
        lse = torch.logsumexp(logits, dim=-1)  # [T]
        return self.zloss_beta * torch.mean(lse * lse)

    def route(self, x: torch.Tensor) -> RoutingOutput:
        """
        x: [T, H] float
        """
        T, H = x.shape
        E, K = self.num_experts, self.k

        # 1) Tính logits
        logits = self.gate(x)  # [T, E]
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # 2) Chọn Top-K expert cho mỗi token
        topk_val, topk_idx = torch.topk(logits, K, dim=-1)  # [T, K] mỗi hàng giảm dần

        # 3) Gating weights = softmax trên K logits đã chọn
        gw = F.softmax(topk_val, dim=-1)  # [T, K], mỗi hàng ~ 1.0

        # 4) Auxiliary losses
        aux_lb = self._compute_aux_lb(topk_idx, gw, E)  # load-balance loss
        aux_z = self._compute_zloss(logits)             # z-loss nếu bật
        aux = aux_lb + aux_z

        # 5) Diagnostics
        with torch.no_grad():
            # utilization (tần suất chọn) theo expert
            util = torch.zeros(E, device=x.device, dtype=gw.dtype)
            util.index_add_(0, topk_idx.reshape(-1), torch.ones_like(gw).reshape(-1))
            util = util / float(T * K)  # tỷ lệ trên tổng T*K slot

            # entropy trung bình trên phân phối gating mỗi token (trên K)
            # (đổi log base e)
            entropy = (-gw * (gw.clamp_min(1e-12)).log()).sum(dim=-1).mean()  # scalar

            diagnostics: Dict[str, torch.Tensor] = {
                "gate_logits_std": logits.std().detach(),
                "utilization": util.detach(),   # [E]
                "entropy": entropy.detach(),    # scalar
                "aux_lb": aux_lb.detach(),
                "aux_z": aux_z.detach(),
            }

        return RoutingOutput(
            dispatch_indices=topk_idx,      # [T, K] long
            gating_weights=gw,               # [T, K] float
            aux_loss=aux,                    # scalar
            positions=None,                  # Dispatcher sẽ tự gán khi áp capacity
            diagnostics=diagnostics,
        )


class NoisyTopKRouter(TopKRouter):
    """
    Noisy Top-K: thêm nhiễu Gaussian nhỏ vào logits để:
      - khuyến khích khám phá sớm (tránh expert chết),
      - ổn định (regularization) nhẹ cho router.

    Tham số bổ sung:
      noisy_std : độ lệch chuẩn Gaussian noise (ví dụ 1/sqrt(E) hoặc 1/E)
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        k: int = 1,
        alpha: float = 1e-2,
        *,
        temperature: float = 1.0,
        zloss_beta: float = 0.0,
        noisy_std: float = 0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            k=k,
            alpha=alpha,
            temperature=temperature,
            zloss_beta=zloss_beta,
        )
        self.noisy_std = float(noisy_std)

    def route(self, x: torch.Tensor) -> RoutingOutput:
        # giống TopKRouter nhưng cộng noise trước khi topk
        T, H = x.shape
        E, K = self.num_experts, self.k

        logits = self.gate(x)  # [T, E]
        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.noisy_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noisy_std

        topk_val, topk_idx = torch.topk(logits, K, dim=-1)  # [T, K]
        gw = F.softmax(topk_val, dim=-1)                    # [T, K]

        aux_lb = self._compute_aux_lb(topk_idx, gw, E)
        aux_z = self._compute_zloss(logits)
        aux = aux_lb + aux_z

        with torch.no_grad():
            util = torch.zeros(E, device=x.device, dtype=gw.dtype)
            util.index_add_(0, topk_idx.reshape(-1), torch.ones_like(gw).reshape(-1))
            util = util / float(T * K)
            entropy = (-gw * (gw.clamp_min(1e-12)).log()).sum(dim=-1).mean()

            diagnostics = {
                "gate_logits_std": logits.std().detach(),
                "utilization": util.detach(),
                "entropy": entropy.detach(),
                "aux_lb": aux_lb.detach(),
                "aux_z": aux_z.detach(),
                "noisy_std": torch.tensor(self.noisy_std, device=x.device),
            }

        return RoutingOutput(
            dispatch_indices=topk_idx,
            gating_weights=gw,
            aux_loss=aux,
            positions=None,
            diagnostics=diagnostics,
        )
