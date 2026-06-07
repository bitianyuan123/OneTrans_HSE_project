import math

from torch import nn
import torch

from onetrans.nn.ffn.hetero_ffn import HeterogeneousFFN


class HeterogeneousAttentionLayer(nn.Module):
    """
    Один слой: heterogeneous multi-head self-attention + индивидуальные FFN.
    Для каждого признака свои Q,K,V.
    """
    def __init__(self, num_features: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Для каждого признака свои Q, K, V (размер d_model x d_k)
        self.q_projs = nn.ParameterList([nn.Parameter(torch.empty(d_model, self.d_k)) for _ in range(num_features)])
        self.k_projs = nn.ParameterList([nn.Parameter(torch.empty(d_model, self.d_k)) for _ in range(num_features)])
        self.v_projs = nn.ParameterList([nn.Parameter(torch.empty(d_model, self.d_v)) for _ in range(num_features)])

        # Индивидуальные FFN
        self.ffns = nn.ModuleList([HeterogeneousFFN(d_model, d_model * 4) for _ in range(num_features)])

        self.out_proj = nn.Linear(d_model, d_model)   # общая проекция выхода
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.q_projs: nn.init.xavier_uniform_(p)
        for p in self.k_projs: nn.init.xavier_uniform_(p)
        for p in self.v_projs: nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features, d_model]
        B, L, D = x.shape
        assert L == self.num_features

        # Проекции Q,K,V для каждого признака: [B, L, n_heads, d_k] -> [B, n_heads, L, d_k]
        Q = []
        K = []
        V = []
        for i in range(L):
            q = torch.matmul(x[:, i, :], self.q_projs[i])   # [B, d_k]
            k = torch.matmul(x[:, i, :], self.k_projs[i])
            v = torch.matmul(x[:, i, :], self.v_projs[i])
            q = q.view(B, self.n_heads, self.d_k)
            k = k.view(B, self.n_heads, self.d_k)
            v = v.view(B, self.n_heads, self.d_v)
            Q.append(q)
            K.append(k)
            V.append(v)
        Q = torch.stack(Q, dim=2)  # [B, n_heads, L, d_k]
        K = torch.stack(K, dim=2)
        V = torch.stack(V, dim=2)

        # Attention scores
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_heads, L, L]
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum по value
        out = torch.matmul(attn, V)  # [B, n_heads, L, d_v]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, d_model]
        out = self.out_proj(out)
        x = x + out
        x = self.norm_attn(x)

        # Индивидуальные FFN
        ffn_out = torch.stack([self.ffns[i](x[:, i, :]) for i in range(L)], dim=1)
        x = x + ffn_out
        x = self.norm_ffn(x)
        return x