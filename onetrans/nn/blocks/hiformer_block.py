import math

from torch import nn
import torch

from onetrans.nn.ffn.hetero_ffn import HeterogeneousFFN


class HiformerLayer(nn.Module):
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

        # Query проекции для каждой головы и каждого признака
        # Размер: [num_features, n_heads, d_model, d_k]
        self.q_projs = nn.Parameter(torch.empty(num_features, n_heads, d_model, self.d_k))

        # Composite projections для каждой головы (low-rank)
        # K̂ʰ = L_kʰ @ R_kʰᵀ
        # L_kʰ: [L*d_model, rank_k], R_kʰ: [L*d_k, rank_k]
        # Но проще через Linear слои
        self.k_low = nn.Parameter(torch.empty(n_heads, d_model * num_features, rank_k))
        self.k_high = nn.Parameter(torch.empty(n_heads, rank_k, self.d_k * num_features))

        self.v_low = nn.Parameter(torch.empty(n_heads, d_model * num_features, rank_v))
        self.v_high = nn.Parameter(torch.empty(n_heads, rank_v, self.d_v * num_features))

        # Индивидуальные FFN для каждого признака
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_features)
        ])

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_projs)
        for h in range(self.n_heads):
            nn.init.xavier_uniform_(self.k_low[h])
            nn.init.xavier_uniform_(self.k_high[h])
            nn.init.xavier_uniform_(self.v_low[h])
            nn.init.xavier_uniform_(self.v_high[h])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # 1. Composite keys для каждой головы
        x_flat = x.view(B, L * D)  # [B, L*D]

        K_composite = []
        V_composite = []

        for h in range(self.n_heads):
            # Key composite: [B, L*D] -> [B, rank_k] -> [B, L*d_k]
            k_mid = torch.matmul(x_flat, self.k_low[h])  # [B, rank_k]
            k_comp = torch.matmul(k_mid, self.k_high[h])  # [B, L*d_k]
            K_composite.append(k_comp)

            # Value composite
            v_mid = torch.matmul(x_flat, self.v_low[h])  # [B, rank_v]
            v_comp = torch.matmul(v_mid, self.v_high[h])  # [B, L*d_v]
            V_composite.append(v_comp)

        # Stack по головам: [B, n_heads, L, d_k] и [B, n_heads, L, d_v]
        K_composite = torch.stack(K_composite, dim=1).view(B, self.n_heads, L, self.d_k)
        V_composite = torch.stack(V_composite, dim=1).view(B, self.n_heads, L, self.d_v)

        # 2. Запросы для каждой головы и каждого признака (простой цикл)
        # q: [B, n_heads, L, d_k]
        q_list = []
        for h in range(self.n_heads):
            q_h_list = []
            for i in range(L):
                # x[:, i, :]: [B, D]
                # self.q_projs[i, h]: [D, d_k]
                q_i = torch.matmul(x[:, i, :], self.q_projs[i, h])  # [B, d_k]
                q_h_list.append(q_i)
            # q_h: [B, L, d_k]
            q_h = torch.stack(q_h_list, dim=1)
            q_list.append(q_h)

        # q: [B, n_heads, L, d_k]
        q = torch.stack(q_list, dim=1)

        # 3. Attention scores
        attn_logits = torch.matmul(q, K_composite.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # 4. Apply attention to composite values
        out = torch.matmul(attn, V_composite)  # [B, n_heads, L, d_v]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D]
        out = self.out_proj(out)
        x = x + out
        x = self.norm_attn(x)

        # 5. Индивидуальные FFN
        ffn_out = torch.stack([self.ffns[i](x[:, i, :]) for i in range(L)], dim=1)
        x = x + ffn_out
        x = self.norm_ffn(x)

        return x