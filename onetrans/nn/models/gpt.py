import torch
from torch import nn

from onetrans.nn.blocks.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList()
        dropout = dropout if self.training else 0.0
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(d_model)
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.dropout(x)
        for block in self.blocks:
            out = block(out, mask)
        out = self.ln(out)
        return out