import torch
from torch import nn

from onetrans.nn.blocks.hiformer_block import HiformerLayer
from onetrans.nn.encoders.categorial import CategoricalEncoder
from onetrans.nn.encoders.multivalent import MultivalentEncoder
from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder


class Hiformer(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_artists: int,
        num_albums: int,
        dense_train_df,
        embedding_size: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_dense_embeddings: int = 8,
        num_layers: int = 2,
        rank_k: int = 128,
        rank_v: int = 1024,
        use_pruning_last: bool = True,
        dropout: float = 0.1,
        n_bins: int = 32,
        train_df_slice: int = 1_000_000,
        output_size: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_dense_embeddings = n_dense_embeddings
        self.use_pruning_last = use_pruning_last

        self.user_encoder = CategoricalEncoder(nn.Embedding(num_users, embedding_size))
        self.item_encoder = CategoricalEncoder(nn.Embedding(num_items, embedding_size))
        self.artist_encoder = MultivalentEncoder(nn.Embedding(num_artists, embedding_size))
        self.album_encoder = MultivalentEncoder(nn.Embedding(num_albums, embedding_size))
        self.piecewise_encoder = PiecewiseLinearEncoder.from_dataset(
            dense_train_df, n_bins, train_df_slice
        )
        dense_raw_dim = sum(self.piecewise_encoder.n_bins)

        self.dense_aggregator = nn.Sequential(
            nn.Linear(dense_raw_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_dense_embeddings * d_model)
        )

        self.user_proj = nn.Linear(embedding_size, d_model)
        self.item_proj = nn.Linear(embedding_size, d_model)
        self.artist_proj = nn.Linear(embedding_size, d_model)
        self.album_proj = nn.Linear(embedding_size, d_model)

        self.task_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 1 task token + 1 user + 1 item + 1 artist + 1 album + n_dense_embeddings
        self.num_features = 1 + 1 + 1 + 1 + 1 + n_dense_embeddings

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(HiformerLayer(
                num_features=self.num_features,
                d_model=d_model,
                n_heads=n_heads,
                rank_k=rank_k,
                rank_v=rank_v,
                dropout=dropout
            ))
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        user_emb = self.user_encoder(inputs["sparse_features"]["uid"])
        item_emb = self.item_encoder(inputs["sparse_features"]["item_id"])
        artist_emb = self.artist_encoder(
            inputs["multivalent_features"]["artist_ids"]["values"],
            inputs["multivalent_features"]["artist_ids"]["lengths"]
        )
        album_emb = self.album_encoder(
            inputs["multivalent_features"]["album_ids"]["values"],
            inputs["multivalent_features"]["album_ids"]["lengths"]
        )
        dense_raw = self.piecewise_encoder(inputs["dense_features"])                   # [B, dense_raw_dim]

        dense_emb = self.dense_aggregator(dense_raw)                                   # [B, n_dense * d_model]
        dense_emb = dense_emb.view(-1, self.n_dense_embeddings, self.d_model)        # [B, n_dense, d_model]

        user_emb = self.user_proj(user_emb).unsqueeze(1)                               # [B, 1, d_model]
        item_emb = self.item_proj(item_emb).unsqueeze(1)
        artist_emb = self.artist_proj(artist_emb).unsqueeze(1)
        album_emb = self.album_proj(album_emb).unsqueeze(1)

        # 4. Формируем последовательность: [task_token, user, item, artist, album, ...dense_embeddings...]
        task = self.task_token.expand(user_emb.size(0), -1, -1)                        # [B, 1, d_model]
        seq = torch.cat([task, user_emb, item_emb, artist_emb, album_emb, dense_emb], dim=1)  # [B, L, d_model]

        for layer in self.layers:
            seq = layer(seq)

        task_out = seq[:, 0, :]   # [B, d_model]

        logits = self.output_layer(task_out)
        return logits

