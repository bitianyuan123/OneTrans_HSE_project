from typing import Dict

import torch
from torch import nn

from onetrans.nn.models.gpt import GPT
from onetrans.utils.create_masked_tensor import create_masked_tensor


class UserEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_seq_len: int = 100,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_items)
        self.pos_embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=max_seq_len)
        self.gpt_encoder = GPT(
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            d_model=embedding_dim
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        history_embedding = self.item_embeddings(inputs['history'])
        padded_tensor, mask = create_masked_tensor(history_embedding, inputs['length'])

        B, S, D = padded_tensor.shape
        positions = torch.arange(S, device=padded_tensor.device)
        pos_embeddings = self.pos_embedding(positions)
        padded_tensor = padded_tensor + pos_embeddings.unsqueeze(0)

        padded_tensor[~mask] = 0
        encoded_seq = self.gpt_encoder(padded_tensor, mask)
        return encoded_seq[mask]
