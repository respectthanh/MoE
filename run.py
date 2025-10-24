import torch
from core.routers.topk import TopKRouter
from core.layer.moe_layer import MoELayer
from core.experts.registry import build_experts

E, K, H = 8, 1, 512
router = TopKRouter(d_model=H, num_experts=E, k=K, alpha=1e-2, zloss_beta=1e-3)
experts = build_experts(["swiglu"] * E, d_model=H, d_ff=int(3.2*H), dropout=0.0)

moe = MoELayer(router, experts, capacity_factor=1.5, drop_tokens=True, batch_first=False)

x = torch.randn(128, H)         # SwiGLU
expert = SwiGLUExpert(d_model=16, d_ff=51)
x = torch.randn(32, 16)  # [T=32, H=16]
y = expert(x)            # [32, 16]

# LoRA
expert = LoRAExpert(d_model=16, d_ff=51, r=8, alpha=16)
y = expert(x)            # Same output shape       # T=128
y, aux = moe(x)
print(y.shape, aux["dropped_tokens"])
