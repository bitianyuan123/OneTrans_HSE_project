from typing import Dict, Any

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from onetrans.nn.encoders.user_history_encoder import UserEncoder


class TrainNIPModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_negatives: int,
        q_counts: torch.Tensor,
        max_seq_len: int = 100,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.encoder = UserEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        self.num_negatives = num_negatives
        self.eps = eps

        q_counts = q_counts.detach().float().clamp_min(0)
        self.register_buffer("q_counts", q_counts)
        self.init_weights(0.02)

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
         for key, value in self.named_parameters():
            if "weight" in key:
                nn.init.trunc_normal_(
                    value.data,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range,
                )
            elif "bias" in key:
                nn.init.zeros_(value.data)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        encoder_output = self.encoder(inputs)
        loss = self.compute_loss(encoder_output, inputs)
        return loss

    def compute_loss(
        self, encoder_output: torch.Tensor, inputs: Dict[str, Any]
    ) -> torch.Tensor:
        pos_ids = inputs['targets']
        pos_emb = self.encoder.item_embeddings(pos_ids)
        pos_logits = (encoder_output * pos_emb).sum(dim=-1)

        batch_size = pos_ids.shape[0]
        random_indices = torch.randint(0, batch_size, (batch_size, self.num_negatives), device=encoder_output.device)
        neg_ids = pos_ids[random_indices]
        neg_emb = self.encoder.item_embeddings(neg_ids)
        neg_logits = (encoder_output.unsqueeze(1) * neg_emb).sum(dim=-1)
        logq = torch.log(self.q_counts + self.eps) - torch.log(self.q_counts.sum() + self.eps)

        neg_logits = neg_logits - logq[neg_ids]
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        return cross_entropy(logits, labels)


class EvalNIPModel(nn.Module):
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
        self.encoder = UserEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        encoder_output = self.encoder(inputs)
        cumulative_lengths = torch.cumsum(inputs['length'], dim=0)
        last_indices = cumulative_lengths - 1
        last_hidden_states = encoder_output[last_indices]

        scores = last_hidden_states @ self.encoder.item_embeddings.weight.T
        return scores