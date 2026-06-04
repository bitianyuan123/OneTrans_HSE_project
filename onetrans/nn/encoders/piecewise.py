import torch
from torch import nn


class PiecewiseLinearEncoder(nn.Module):
    @staticmethod
    def compute_bins(
            X: torch.Tensor,
            n_bins: int,
    ) -> list[torch.Tensor]:
        bins_list = []
        for i in range(X.shape[1]):
            col = X[:, i]
            quantiles = torch.quantile(col, torch.linspace(0.0, 1.0, n_bins + 1, device=col.device))
            unique_bins = torch.unique(quantiles)
            bins_list.append(unique_bins)
        return bins_list

    @classmethod
    def from_dataset(cls, dense_train_df, n_bins=32, train_df_slice: int = 1_000_000):
        if len(dense_train_df) > train_df_slice:
            df_slice = dense_train_df.slice(0, train_df_slice)
        else:
            df_slice = dense_train_df

        X = df_slice.to_torch().to(torch.float32)
        bins_list = cls.compute_bins(X, n_bins)

        n_features = X.shape[1]
        max_n_bins = max(len(b) - 1 for b in bins_list)

        weight = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
        bias = torch.zeros(n_features, max_n_bins, dtype=torch.float32)

        n_bins_per_feature = []
        single_bin_flags = []

        for i, bins in enumerate(bins_list):
            n_bin = len(bins) - 1
            assert n_bin >= 1, "There is a column with only one unique value"

            n_bins_per_feature.append(n_bin)
            single_bin_flags.append(n_bin == 1)

            for j in range(n_bin):
                a = bins[j]
                b = bins[j + 1]
                if abs(b - a) < 1e-8:
                    w = 1.0
                    c = 0.0
                else:
                    w = 1.0 / (b - a)
                    c = -a * w
                weight[i, j] = w
                bias[i, j] = c

        if all(nb == max_n_bins for nb in n_bins_per_feature):
            mask = None
        else:
            mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
            for i, nb in enumerate(n_bins_per_feature):
                mask[i, :nb] = True

        single_bin_mask = torch.tensor(single_bin_flags, dtype=torch.bool)

        return cls(weight, bias, mask, n_bins_per_feature, single_bin_mask)

    def __init__(self, weight, bias, mask, n_bins, single_bin_mask):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("single_bin_mask", single_bin_mask)
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        self._n_bins = n_bins

    @property
    def n_bins(self):
        return self._n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        max_n_bins = self.weight.shape[1]

        x_exp = x.unsqueeze(-1).expand(-1, -1, max_n_bins)

        out = self.bias + self.weight * x_exp

        out[:, :, 0] = out[:, :, 0].clamp(max=1.0)
        if max_n_bins > 1:
            out[:, :, 1:-1] = out[:, :, 1:-1].clamp(0.0, 1.0)
            out[:, :, -1] = out[:, :, -1].clamp(min=0.0)

        if self.single_bin_mask.any():
            single_mask = self.single_bin_mask.unsqueeze(0).unsqueeze(-1)
            single_val = (x.unsqueeze(-1) >= self.bias[:, :1]).float()
            out = torch.where(single_mask, single_val, out)

        out = out.reshape(batch_size, -1)

        if self.mask is not None:
            out = out[:, self.mask.view(-1)]

        return out
