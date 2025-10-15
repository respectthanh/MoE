import torch
from torch.utils.data import Dataset

class ToyAutoEncoderDataset(Dataset):
    """
    Nhiệm vụ: tái tạo lại input (reconstruction).
    - Sinh token embedding ngẫu nhiên quanh vài 'cụm' (miền), giúp router có dấu hiệu phân tách.
    - MoE học: expert khác nhau xử lý cụm khác nhau; LB loss giúp chia tải đều.
    """
    def __init__(self, n_seq=1024, T=64, d_model=64, n_clusters=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X = []
        centers = torch.randn(n_clusters, d_model, generator=g) * 2.0  # tâm cụm
        for _ in range(n_seq):
            # mỗi sequence: chọn ngẫu nhiên 1 tâm cho mỗi token
            cluster_ids = torch.randint(0, n_clusters, (T,), generator=g)
            x = centers[cluster_ids] + 0.5*torch.randn(T, d_model, generator=g)
            self.X.append(x)
        self.X = torch.stack(self.X, dim=0)  # [N, T, D]
        self.Y = self.X.clone()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
