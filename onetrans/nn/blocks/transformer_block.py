from typing import Optional

import torch
from torch import nn

from onetrans.nn.attention.causal_attention import CausalSelfAttention
from onetrans.nn.ffn.ffn import FFN


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        dropout = dropout if self.training else 0.0
        self.causal_self_attention = CausalSelfAttention(
            d_model=d_model,
            dropout=dropout,
            n_heads=n_heads,
        )
        self.mlp = FFN(d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            B, S, _ = x.shape
            mask = torch.ones(B, S, dtype=torch.bool)
        residual = x
        out = self.ln1(x)
        out = self.causal_self_attention(out, mask)
        out = self.dropout(out)
        out = out + residual

        residual = out
        out = self.ln2(out)
        out = self.mlp(out)
        out = self.dropout(out)
        out = out + residual

        return out