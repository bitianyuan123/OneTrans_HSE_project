from onetrans.nn.attention.mixed_attention import MixedCausalSelfAttention
from onetrans.nn.ffn.mixed_ffn import MixedFFN

import torch
import torch.nn as nn


class CoreOneTransBlock(nn.Module):
    def __init__(
        self,
        max_seq_len : int = 128,
        d_model : int = 256,
        n_heads : int = 8,
        ns_tokens_num : int = 8,
        dropout : float = 0.0,
        out_seq_num : int = 60
    ):
        super().__init__()
        self.norm = nn.RMSNorm(normalized_shape=(d_model, ))
        self.out_seq_num = out_seq_num
        self.ns_tokens_num = ns_tokens_num
        self.mixed_attn = MixedCausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            ns_tokens_num=ns_tokens_num,
            dropout=dropout
        )
        self.mixed_ffn = MixedFFN(
            d_model=d_model,
            ns_tokens_num=ns_tokens_num,
            dropout=dropout
        )

    def forward(self, x, mask):
        '''
        args:
            x : inputs of shape (batch_size, S + NS, d_model)
            mask : boolean mask of shape (batch_size, S + NS)
        returns:
            out : outputs of shape (batch_size, out_seq_num + NS, d_model)
            mask : updated mask of shape (batch_size, out_seq_num + NS)
        '''
        z = self.mixed_attn(self.norm(x), mask=mask) + x
        z = z + self.mixed_ffn(self.norm(z))

        # pyramid 缩层：保留「最新（尾部）」的 S token（论文 §3.4 的 tail index set
        # Q = {S_in - out_seq_num + 1, ..., S_in}），而非最旧（头部）。配合左 padding。
        s_in = x.shape[1] - self.ns_tokens_num
        s_tokens = z[:, s_in - self.out_seq_num : s_in, :]
        ns_tokens = z[:, -self.ns_tokens_num:, :]
        z = torch.cat([s_tokens, ns_tokens], dim=1)

        s_mask = mask[:, s_in - self.out_seq_num : s_in]
        ns_mask = mask[:, -self.ns_tokens_num:]
        mask = torch.cat([s_mask, ns_mask], dim=1)

        return z, mask
