# moe/core/layer/moe_layer.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.core.routers.base import BaseRouter, RoutingOutput
from moe.core.runtime.dispatcher import simple_dispatch
from moe.core.runtime.combiner import simple_combine


class MoELayer(nn.Module):
    """
    MoELayer: Bộ điều phối trung tâm của khung Mixture-of-Experts.
    -------------------------------------
    Nhiệm vụ:
      1. Gọi router để lấy quyết định định tuyến token -> expert
      2. Phân phối token đến các expert tương ứng (Dispatcher)
      3. Gọi tính toán của expert (song song)
      4. Gom kết quả về thứ tự gốc (Combiner)
      5. Trả về kết quả + aux_outputs để vòng lặp huấn luyện dùng

    Đầu vào:
      inputs: Tensor [B, S, H] hoặc [T, H]

    Đầu ra:
      output: Tensor [B, S, H]
      aux_outputs: {
          'aux_loss': Tensor scalar,
          'dropped_tokens': float,
          'diagnostics': Dict[str, Tensor]
      }
    """

    def __init__(
        self,
        router: BaseRouter,
        experts: nn.ModuleList,
        *,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.num_experts = len(experts)
        self.capacity_factor = float(capacity_factor)
        self.drop_tokens = drop_tokens
        self.batch_first = batch_first

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Flatten input nếu cần
        if self.batch_first:
            B, S, H = inputs.shape
            x = inputs.reshape(B * S, H)  # [T, H]
        else:
            x = inputs
            B, S, H = 1, x.shape[0], x.shape[1]

        T = x.size(0)

        # Route: router trả về RoutingOutput
        routing_output: RoutingOutput = self.router(x)
        topk_idx = routing_output.dispatch_indices  # [T, K]
        gating_weights = routing_output.gating_weights  # [T, K]
        aux_loss = routing_output.aux_loss
        diagnostics = routing_output.diagnostics or {}

        # Dispatch: gom token theo expert
        dispatched_inputs, dispatch_info, dropped_tokens = simple_dispatch(
            x, topk_idx, capacity_factor=self.capacity_factor, drop_tokens=self.drop_tokens
        )
        # dispatched_inputs: list[E] mỗi phần [C, H] (C = capacity)
        # mask: list[E] same shape, cho biết vị trí có token thật
        # dropped_tokens: float tỉ lệ bị drop

        # Expert compute: chạy từng expert trên batch tương ứng
        expert_outputs: List[torch.Tensor] = []
        for i, expert in enumerate(self.experts):
            inp = dispatched_inputs[i]  # [C, H]
            if inp.numel() == 0:
                # expert này không có token nào
                out = torch.zeros_like(inp)
            else:
                out = expert(inp)  # [C, H]
            expert_outputs.append(out)

        # Combine: gom output về thứ tự token ban đầu
        combined_output = simple_combine(
            expert_outputs, topk_idx, gating_weights, dispatch_info, T
        )  # [T, H]

        # Reshape lại [B, S, H]
        output = combined_output.reshape(B, S, H)

        # Gói aux_outputs
        aux_outputs = {
            "aux_loss": aux_loss,
            "dropped_tokens": dropped_tokens,
            "diagnostics": diagnostics,
        }

        return output, aux_outputs
