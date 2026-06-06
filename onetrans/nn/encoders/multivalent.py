import torch
from torch import nn
from torch.nn.functional import embedding_bag


class MultivalentEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        if values.dim() == 1:
            values = values.unsqueeze(1)
        num_hashes = values.shape[1]
        offsets = torch.zeros(batch_size, dtype=torch.long, device=values.device)
        offsets[1:] = lengths.cumsum(dim=0)[:-1]
        outputs = []
        for h in range(num_hashes):
            emb = embedding_bag(
                self.embeddings.weight,
                values[:, h].contiguous(),
                offsets=offsets,
                mode="mean",
                sparse=False,
            )
            outputs.append(emb)
        return torch.cat(outputs, dim=1)
