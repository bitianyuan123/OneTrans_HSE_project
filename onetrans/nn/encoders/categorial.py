import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(ids)
