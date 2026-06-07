import torch
from torch import nn

from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder
class YambdaEmbedder(nn.Module):
    def __init__(
        self,
        item_embedding: nn.Module,
        user_embedding: nn.Embedding,
        artist_embedding: nn.Module,
        album_embedding: nn.Module,
        piecewise_encoder: PiecewiseLinearEncoder,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.item_embedding = item_embedding
        self.user_embedding = user_embedding
        self.artist_embedding = artist_embedding
        self.album_embedding = album_embedding
        self.piecewise_encoder = piecewise_encoder
        self.max_seq_len = max_seq_len

    @property
    def embed_dim(self) -> int:
        emb = self.item_embedding
        # nn.Embedding has .embedding_dim; MultihashEmbedding wraps one internally
        return emb.embedding_dim if hasattr(emb, "embedding_dim") else emb.embedding.embedding_dim

    @property
    def seq_in_dim(self) -> int:
        return self.embed_dim

    @property
    def ns_group_dims(self) -> list[int]:
        d = self.embed_dim
        return [
            self.piecewise_encoder.out_features,  # dense
            d,  # uid
            d,  # item_id
            d,  # artist_ids
            d,  # album_ids
        ]

    def forward(self, batch: dict) -> dict:
        flat_ids = batch["S"]["item_id"]
        flat_ts = batch["S"]["timestamps"]
        lengths = batch["S"]["lengths"]

        sequences = flat_ids.split(lengths.tolist())
        padded_ids = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        padded_ids = padded_ids[:, : self.max_seq_len]

        ts_sequences = flat_ts.split(lengths.tolist())
        padded_ts = nn.utils.rnn.pad_sequence(ts_sequences, batch_first=True)
        padded_ts = padded_ts[:, : self.max_seq_len]

        L = padded_ids.shape[1]
        seq_mask = (
            torch.arange(L, device=padded_ids.device).unsqueeze(0)
            < lengths.unsqueeze(1).clamp(max=self.max_seq_len)
        )
        seq_emb = self.item_embedding(padded_ids) * seq_mask.unsqueeze(-1)

        ns = batch["NS"]
        uid_emb = self.user_embedding(ns["sparse_features"]["uid"])
        item_emb = self.item_embedding(ns["sparse_features"]["item_id"])
        dense_enc = self.piecewise_encoder(ns["dense_features"])

        artist = ns["multivalent_features"]["artist_ids"]
        album = ns["multivalent_features"]["album_ids"]
        artist_emb = self.artist_embedding(artist["values"], artist["lengths"])
        album_emb = self.album_embedding(album["values"], album["lengths"])

        return {
            "seq_features": [seq_emb],
            "seq_masks": [seq_mask],
            "seq_timestamps": [padded_ts],
            "ns_groups": [dense_enc, uid_emb, item_emb, artist_emb, album_emb],
        }
