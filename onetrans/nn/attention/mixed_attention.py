import torch
from torch import nn
import torch.nn.functional as F


class MixedCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ns_tokens_num: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ns_tokens_num = ns_tokens_num
        self.dropout = dropout
        self.head_dim = d_model // n_heads

        # под все s токены заводится единая проекция
        self.W_s = nn.Linear(d_model, 3 * d_model, bias=False)

        # для каждого ns токена заводится своя проекция (см схему из статьи)
        self.W_ns_list = nn.ModuleList([
            nn.Linear(d_model, 3 * d_model, bias=False)
            for _ in range(ns_tokens_num)
        ])

        self.final_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_mask(self, mask: torch.Tensor, total_len: int) -> torch.Tensor:
        B = mask.shape[0]
        device = mask.device

        padding_mask = ~mask
        padding_mask = padding_mask[:, None, None, :]

        causal_mask = torch.triu(
            torch.ones(total_len, total_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        causal_mask = causal_mask[None, None, :, :]

        combined_mask = torch.zeros(B, 1, total_len, total_len, device=device)
        combined_mask = combined_mask.masked_fill(padding_mask, float('-inf'))
        combined_mask = combined_mask.masked_fill(causal_mask, float('-inf'))

        return combined_mask

    def forward(
            self,
            x: torch.Tensor,
            sequential_tokens_num: int,
            mask: torch.Tensor
    ) -> torch.Tensor:
        B, total_len, D = x.shape
        ns_len = total_len - sequential_tokens_num

        # по статье все QKV в итоге идут в один multi head attn
        Q = torch.zeros(B, total_len, self.n_heads, self.head_dim, device=x.device)
        K = torch.zeros(B, total_len, self.n_heads, self.head_dim, device=x.device)
        V = torch.zeros(B, total_len, self.n_heads, self.head_dim, device=x.device)

        # sequential часть
        s_tokens = x[:, :sequential_tokens_num, :] # (B, s_num, D)
        s_qkv = self.W_s(s_tokens) # (B, s_num, 3 * d_model)
        s_qkv = s_qkv.reshape(B, sequential_tokens_num, 3, self.n_heads, self.head_dim)
        s_Q, s_K, s_V = s_qkv.unbind(dim=2)

        Q[:, :sequential_tokens_num] = s_Q
        K[:, :sequential_tokens_num] = s_K
        V[:, :sequential_tokens_num] = s_V

        # non sequential часть
        ns_tokens = x[:, sequential_tokens_num:, :]
        for i in range(ns_len):
            ns_token = ns_tokens[:, i, :]  # (B, D)
            ns_qkv = self.W_ns_list[i](ns_token)  # (B, 3 * D)
            ns_qkv = ns_qkv.reshape(B, 3, self.n_heads, self.head_dim)
            ns_Q, ns_K, ns_V = ns_qkv.unbind(dim=1)

            Q[:, sequential_tokens_num + i] = ns_Q
            K[:, sequential_tokens_num + i] = ns_K
            V[:, sequential_tokens_num + i] = ns_V

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_mask = self._build_mask(mask=mask, total_len=total_len)
        scores = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        assert scores.shape == (B, self.n_heads, total_len, self.head_dim)

        scores = scores.transpose(1, 2).reshape(B, total_len, D)
        scores = self.final_proj(scores)
        return scores