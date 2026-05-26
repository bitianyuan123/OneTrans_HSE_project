import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = d_model // n_heads

        self.W = nn.Linear(d_model, 3 * d_model, bias=False)

        self.final_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_mask(self, x: torch.Tensor, mask) -> torch.Tensor:
        B, S, D = x.shape
        device = x.device

        padding_mask = ~mask
        padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, S)

        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, S, S)

        combined_mask = torch.zeros(B, 1, S, S, device=device)
        combined_mask = combined_mask.masked_fill(padding_mask, float('-inf'))
        combined_mask = combined_mask.masked_fill(causal_mask, float('-inf'))

        return combined_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.W(x)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        Q, K, V = qkv.unbind(dim=2)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_mask = self._build_mask(x, mask)
        scores = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        assert scores.shape == (B, self.n_heads, S, self.head_dim)

        scores = scores.transpose(1, 2).reshape(B, S, D)
        scores = self.final_proj(scores)
        return scores