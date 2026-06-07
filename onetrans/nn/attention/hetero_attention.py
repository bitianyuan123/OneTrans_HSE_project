import math

from torch import nn
import torch


class HeterogeneousAttentionLayer(nn.Module):
    def __init__(self, num_features: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Query, Key, Value проекции
        # Reshape для векторизованного matmul
        self.q_projs = nn.Parameter(torch.empty(num_features, n_heads, d_model, self.d_k))
        self.k_projs = nn.Parameter(torch.empty(num_features, n_heads, d_model, self.d_k))
        self.v_projs = nn.Parameter(torch.empty(num_features, n_heads, d_model, self.d_v))

        self.out_proj = nn.Linear(d_model, d_model)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_features)
        ])

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_projs)
        nn.init.xavier_uniform_(self.k_projs)
        nn.init.xavier_uniform_(self.v_projs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Reshape для batch matrix multiplication
        # x: [B, L, D] -> [B, L, 1, D] для broadcasting
        x_expanded = x.unsqueeze(2)  # [B, L, 1, D]

        # Q: [B, L, n_heads, d_k]
        Q = torch.matmul(x_expanded, self.q_projs.unsqueeze(0))  # [B, L, n_heads, d_k]
        K = torch.matmul(x_expanded, self.k_projs.unsqueeze(0))
        V = torch.matmul(x_expanded, self.v_projs.unsqueeze(0))

        # Transpose для attention: [B, n_heads, L, d_k]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, n_heads, L, d_v]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D]
        out = self.out_proj(out)
        x = x + out
        x = self.norm_attn(x)

        # FFN
        ffn_out = torch.stack([self.ffns[i](x[:, i, :]) for i in range(L)], dim=1)
        x = x + ffn_out
        x = self.norm_ffn(x)

        return x
