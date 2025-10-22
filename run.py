import torch
from core.routers.topk import TopKRouter
from core.layer.moe_layer import MoELayer

# toy experts
class ToyExpert(torch.nn.Module):
    def __init__(self, h): super().__init__(); self.lin = torch.nn.Linear(h, h, bias=False)
    def forward(self, x): return self.lin(x)

E, K, H = 4, 2, 16
experts = torch.nn.ModuleList([ToyExpert(H) for _ in range(E)])
router  = TopKRouter(d_model=H, num_experts=E, k=K, alpha=1e-2)

moe = MoELayer(router, experts, capacity_factor=1.25, drop_tokens=True, batch_first=False)

T = 32
x = torch.randn(T, H)
y, aux = moe(x)
print(y.shape, aux["dropped_tokens"], list(aux["diagnostics"].keys()))
