from torch import nn
import torch


class FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        dropout_p = dropout if self.training else 0.0
        self.network = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(4 * d_model, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)