import torch
from torch import nn
from torch.nn.functional import softmax
import polars as pl

from onetrans.nn.encoders.categorial import CategoricalEncoder
from onetrans.nn.encoders.multivalent import MultivalentEncoder
from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder


class MixtureLowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.rank = rank

        self.U = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.V = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.bias = nn.Parameter(torch.empty(input_dim))

        self.gate = nn.Linear(input_dim, num_experts)

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # [bs, num_experts, rank]
        xl_v = torch.einsum("bd,edr->ber", xl, self.V)

        # [bs, num_experts, input_dim]
        expert_update = torch.einsum("ber,edr->bed", xl_v, self.U)
        expert_update = expert_update + self.bias
        expert_update = expert_update * x0.unsqueeze(1)

        # [bs, num_experts]
        gate_logits = self.gate(xl)
        gate_weights = softmax(gate_logits, dim=-1)

        expert_update = expert_update * gate_weights.unsqueeze(-1)
        mixed_update = expert_update.sum(dim=1)

        return mixed_update + xl


class MixtureLowRankCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, num_experts: int, rank: int):
        super().__init__()
        self.layers = nn.ModuleList([
            MixtureLowRankCrossLayer(input_dim, num_experts, rank)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x

        for layer in self.layers:
            xl = layer(x0, xl)

        return xl


class DeepNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DCNV2(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            cross_layers: int,
            deep_units: list[int],
            input_size: int,
            dense_train_df: pl.DataFrame,
            n_bins: int,
            train_df_slice: int,
            num_items: int,
            num_users: int,
            num_artists: int,
            num_albums: int,
            num_experts: int = 4,
            low_rank: int = 32,
            output_size: int = 2,
    ):
        super().__init__()

        self.user_encoder = CategoricalEncoder(nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_size
        ))
        self.item_encoder = CategoricalEncoder(nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_size
        ))

        self.artist_encoder = MultivalentEncoder(nn.Embedding(
            num_embeddings=num_artists,
            embedding_dim=embedding_size
        ))

        self.album_encoder = MultivalentEncoder(nn.Embedding(
            num_embeddings=num_albums,
            embedding_dim=embedding_size
        ))

        self.piecewise_encoder = PiecewiseLinearEncoder.from_dataset(
            dense_train_df=dense_train_df,
            n_bins=n_bins,
            train_df_slice=train_df_slice
        )

        self.deep_network = DeepNetwork(
            input_dim=input_size,
            hidden_units=deep_units,
        )
        self.cross_network = MixtureLowRankCrossNetwork(
            input_dim=input_size,
            num_layers=cross_layers,
            num_experts=num_experts,
            rank=low_rank
        )

        self.output_layer = nn.Linear(
            in_features=input_size + (deep_units[-1] if deep_units else input_size),
            out_features=output_size
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        uid_emb = self.user_encoder(inputs["sparse_features"]["uid"])
        item_emb = self.item_encoder(inputs["sparse_features"]["item_id"])

        artist_emb = self.artist_encoder(
            inputs["multivalent_features"]["artist_ids"]["values"],
            inputs["multivalent_features"]["artist_ids"]["lengths"]
        )
        album_emb = self.album_encoder(
            inputs["multivalent_features"]["album_ids"]["values"],
            inputs["multivalent_features"]["album_ids"]["lengths"]
        )

        dense_emb = self.piecewise_encoder(inputs["dense_features"])

        uid_emb = uid_emb.reshape(uid_emb.shape[0], -1)
        item_emb = item_emb.reshape(item_emb.shape[0], -1)

        artist_emb = artist_emb.reshape(artist_emb.shape[0], -1)
        album_emb = album_emb.reshape(album_emb.shape[0], -1)

        features = torch.cat([
            dense_emb,
            uid_emb,
            item_emb,
            artist_emb,
            album_emb
        ], dim=1)

        deep_out = self.deep_network(features)
        cross_out = self.cross_network(features)

        combined = torch.cat([cross_out, deep_out], dim=1)

        output = self.output_layer(combined)
        return output

