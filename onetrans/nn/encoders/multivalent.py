import torch
from torch import nn
from torch.nn.functional import embedding_bag


class MultivalentEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        num_hashes = ids.shape[1]

        offsets = torch.zeros(batch_size, dtype=torch.long, device=ids.device)
        offsets[1:] = lengths.cumsum(dim=0)[:-1]

        outputs = []
        for h in range(num_hashes):
            current_ids = ids[:, h] # [num_values]

            emb = embedding_bag(
                self.embeddings.weight,
                current_ids,
                offsets=offsets,
                mode='mean',
                sparse=False
            ) # [bs, embedding_dim]

            outputs.append(emb)

        # [bs, num_hashes, embedding_dim]
        return torch.stack(outputs, dim=1)
