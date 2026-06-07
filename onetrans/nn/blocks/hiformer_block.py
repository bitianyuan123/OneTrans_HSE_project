import math

from torch import nn
import torch

from onetrans.nn.ffn.hetero_ffn import HeterogeneousFFN


class HiformerLayer(nn.Module):
    """
    Один слой Hiformer (секция 3.4) с low-rank approximation (секция 3.5.1)
    """
    def __init__(self, num_features: int, d_model: int, n_heads: int,
                 rank_k: int, rank_v: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.rank_k = rank_k
        self.rank_v = rank_v

        # Query проекции – индивидуальные для каждого признака (как в HeteroAtt)
        self.q_projs = nn.ParameterList([nn.Parameter(torch.empty(d_model, self.d_k)) for _ in range(num_features)])

        # Composite projections: K_hat = L_k @ R_k^T,  L_k: [L*d_model, rank_k], R_k: [L*d_k, rank_k]
        # но удобнее сделать через линейные слои без явного хранения L*d матриц.
        # Реализуем как две линейные проекции: сначала проецируем concat всех признаков в размер rank,
        # затем проецируем обратно в L*d_k.
        self.k_low = nn.Linear(d_model * num_features, rank_k, bias=False)
        self.k_high = nn.Linear(rank_k, self.d_k * num_features, bias=False)
        self.v_low = nn.Linear(d_model * num_features, rank_v, bias=False)
        self.v_high = nn.Linear(rank_v, self.d_v * num_features, bias=False)

        # Индивидуальные FFN (как в HeteroAtt)
        self.ffns = nn.ModuleList([HeterogeneousFFN(d_model, d_model * 4) for _ in range(num_features)])

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.q_projs: nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.k_low.weight)
        nn.init.xavier_uniform_(self.k_high.weight)
        nn.init.xavier_uniform_(self.v_low.weight)
        nn.init.xavier_uniform_(self.v_high.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        # 1. Composite keys и values
        x_flat = x.view(B, L * D)                     # [B, L*D]
        k_composite = self.k_high(self.k_low(x_flat)) # [B, L*d_k]
        v_composite = self.v_high(self.v_low(x_flat)) # [B, L*d_v]

        # 2. Запросы (индивидуальные)
        Q = []
        for i in range(L):
            q = torch.matmul(x[:, i, :], self.q_projs[i])  # [B, d_k]
            q = q.view(B, self.n_heads, self.d_k)
            Q.append(q)
        Q = torch.stack(Q, dim=2)  # [B, n_heads, L, d_k]

        # 3. Attention: Q * composite_keys^T
        # composite_keys нужно reshape в [B, n_heads, L, d_k]
        k_composite = k_composite.view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, n_heads, L, d_k]
        attn_logits = torch.matmul(Q, k_composite.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_heads, L, L]
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # 4. Применяем attention к composite values
        v_composite = v_composite.view(B, L, self.n_heads, self.d_v).permute(0, 2, 1, 3)  # [B, n_heads, L, d_v]
        out = torch.matmul(attn, v_composite)  # [B, n_heads, L, d_v]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D]
        out = self.out_proj(out)
        x = x + out
        x = self.norm_attn(x)

        # 5. Индивидуальные FFN
        ffn_out = torch.stack([self.ffns[i](x[:, i, :]) for i in range(L)], dim=1)
        x = x + ffn_out
        x = self.norm_ffn(x)
        return x
