import torch
import torch.nn as nn

from onetrans.nn.blocks.onetrans_block import CoreOneTransBlock


class OneTrans(nn.Module):
    def __init__(
        self,
        d_model : int = 256,
        num_blocks : int = 6,
        num_heads : int = 4,
        max_seq_len : int = 100,
        min_seq_len : int = 10,
        ns_tokens_num : int = 8,
        dropout : float = 0.0,
        dimensionality_reduction : str = "linear"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ns_tokens_num = ns_tokens_num
        if dimensionality_reduction == "linear":
            self.dims = torch.linspace(max_seq_len, min_seq_len, num_blocks + 1, dtype=torch.long)
        elif dimensionality_reduction == "exponential":
            self.dims = torch.logspace(max_seq_len, min_seq_len, num_blocks + 1, dtype=torch.long)
        self.blocks = nn.ModuleList([
            CoreOneTransBlock(
                self.dims[i],
                self.d_model,
                self.num_heads,
                self.ns_tokens_num,
                dropout,
                self.dims[i + 1]
            ) for i in range(self.num_blocks)
        ])
        self.linear = nn.Linear(self.d_model, 2)

    def forward(self, x, mask):
        '''
            args:
                x : padded sequence of shape (batch_size, max_seq_len, d_model)
                x : boolean padding mask of shape (batch_size, max_seq_len)
            returns:
                out : logits of shape (batch_size, )
        '''
        for block in self.blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x
