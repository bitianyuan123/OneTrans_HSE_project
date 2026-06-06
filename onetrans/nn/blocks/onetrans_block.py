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
            mask : boolean mask of shape (batch_size, max_seq_len)
        returns:
            out : outputs of shape (batch_size, out_seq_num + NS, d_model)
        '''
        z = self.mixed_attn(self.norm(x), mask=mask) + x
        x = z + self.mixed_ffn(self.norm(z))
        # x of shape (B, input_S + NS, D)
        x = x[:, :self.out_seq_num + self.ns_tokens_num, :]
        # now of desired shape: (B, out_S + NS, D)
        return x
