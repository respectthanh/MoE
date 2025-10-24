# moe/core/experts/registry.py
from __future__ import annotations
from typing import List, Dict, Any
import torch
import torch.nn as nn

from .mlp import SwiGLUExpert, LoRAExpert


def make_expert(kind: str, d_model: int, **kw) -> nn.Module:
    """
    Khởi tạo 1 expert theo 'kind'.
    Các kinds hỗ trợ:
      - "swiglu"        : FFN SwiGLU tiêu chuẩn
      - "swiglu_small"  : d_ff = 2.0 * d_model
      - "swiglu_big"    : d_ff = 4.0 * d_model
      - "lora"          : LoRAExpert(d_ff, r, alpha)
    """
    kind = kind.lower()
    if kind == "swiglu":
        d_ff = int(kw.get("d_ff", int(3.2 * d_model)))
        return SwiGLUExpert(d_model, d_ff, dropout=float(kw.get("dropout", 0.0)))

    if kind == "swiglu_small":
        d_ff = int(2.0 * d_model)
        return SwiGLUExpert(d_model, d_ff, dropout=float(kw.get("dropout", 0.0)))

    if kind == "swiglu_big":
        d_ff = int(4.0 * d_model)
        return SwiGLUExpert(d_model, d_ff, dropout=float(kw.get("dropout", 0.0)))

    if kind == "lora":
        d_ff = int(kw.get("d_ff", int(3.2 * d_model)))
        r = int(kw.get("r", 8))
        alpha = int(kw.get("alpha", 16))
        return LoRAExpert(d_model, d_ff, r=r, alpha=alpha, dropout=float(kw.get("dropout", 0.0)))

    raise ValueError(f"Unknown expert kind: {kind}")


def build_experts(
    kinds: List[str],
    d_model: int,
    **common_kw,
) -> nn.ModuleList:
    """
    Tạo danh sách experts từ danh sách kinds. Ví dụ:
      kinds = ["swiglu_small", "swiglu_big", "swiglu", "lora", ...]
    Các kw chung như d_ff/dropout/r/alpha sẽ được truyền cho từng expert
    """
    experts = []
    for k in kinds:
        experts.append(make_expert(k, d_model, **common_kw))
    return nn.ModuleList(experts)
