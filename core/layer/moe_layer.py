from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.routers.base import BaseRouter, RoutingOutput
from core.runtime.dispatcher import simple_dispatch
from core.runtime.combiner import simple_combine


class MoELayer(nn.Module):
    """
    MoELayer: Bộ điều phối trung tâm của khung Mixture-of-Experts.
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
            x = inputs.reshape(B * S, H)
        else:
            x = inputs
            B, S, H = 1, x.shape[0], x.shape[1]

        T = x.size(0)

        # Route
        routing_output: RoutingOutput = self.router(x)
        topk_idx = routing_output.dispatch_indices
        gating_weights = routing_output.gating_weights
        aux_loss = routing_output.aux_loss
        diagnostics = routing_output.diagnostics or {}

        # Dispatch
        dispatched_inputs, dispatch_info, dropped_tokens = simple_dispatch(
            x, topk_idx, capacity_factor=self.capacity_factor, drop_tokens=self.drop_tokens
        )

        # Expert compute
        expert_outputs: List[torch.Tensor] = []
        for i, expert in enumerate(self.experts):
            inp = dispatched_inputs[i]
            if inp.numel() == 0:
                out = torch.zeros_like(inp)
            else:
                out = expert(inp)
            expert_outputs.append(out)

        # Combine
        combined_output = simple_combine(
            expert_outputs, topk_idx, gating_weights, dispatch_info, T
        )

        # Reshape lại [B, S, H]
        output = combined_output.reshape(B, S, H)

        # Gói aux_outputs
        aux_outputs = {
            "aux_loss": aux_loss,
            "dropped_tokens": dropped_tokens,
            "diagnostics": diagnostics,
        }

        return output, aux_outputs
