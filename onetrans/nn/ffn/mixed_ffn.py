from torch import nn
import torch


class MixedFFN(nn.Module):
    def __init__(self, d_model: int, ns_tokens_num: int, dropout: float = 0.0):
        super().__init__()
        dropout_p = dropout if self.training else 0.0
        self.ns_tokens_num = ns_tokens_num

        self.network_s = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(4 * d_model, d_model, bias=False)
        )
        self.networks_ns_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=False),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(4 * d_model, d_model, bias=False)
            ) for _ in range(ns_tokens_num)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total_len, D = x.shape
        sequential_tokens_num = total_len - self.ns_tokens_num

        # sequential часть
        s_input = x[:, :sequential_tokens_num, :]  # (B, s_num, D)
        sequential_output = self.network_s(s_input)

        # non sequential часть
        ns_input = x[:, sequential_tokens_num:, :]  # (B, ns_num, D)
        non_sequential_output_list = [
            self.networks_ns_list[i](
                ns_input[:, i:i+1, :]  # (B, 1, D)
            ) for i in range(self.ns_tokens_num)
        ]

        return torch.cat([sequential_output] + non_sequential_output_list, dim=1)
