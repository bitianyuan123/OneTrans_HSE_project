import torch
from torch import nn

from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder
from onetrans.nn.tokenizer import embed_multivalent


class YambdaEmbedder(nn.Module):
    def __init__(
        self,
        embedding,
        piecewise_encoder,
        max_seq_len = 100,
        num_hashes = 1,
    ):
        super().__init__()
        self.embedding = embedding
        self.piecewise_encoder = piecewise_encoder
        self.max_seq_len = max_seq_len
        self.num_hashes = num_hashes

    @property
    def seq_in_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def ns_group_dims(self) -> list[int]:
        d = self.embedding.embedding_dim * self.num_hashes
        return [
            self.piecewise_encoder.out_features,  # dense
            d,  # uid
            d,  # item_id
            d,  # artist_ids
            d,  # album_ids
        ]

    def _embed_sparse(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() == 1:
            return self.embedding(ids)
        return self.embedding(ids).reshape(ids.shape[0], -1)

    def forward(self, batch: dict) -> dict:
        flat_ids = batch["S"]["item_id"]
        lengths = batch["S"]["lengths"]

        sequences = flat_ids.split(lengths.tolist())
        padded_ids = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        padded_ids = padded_ids[:, : self.max_seq_len]

        L = padded_ids.shape[1]
        seq_mask = (
            torch.arange(L, device=padded_ids.device).unsqueeze(0)
            < lengths.unsqueeze(1).clamp(max=self.max_seq_len)
        )
        seq_emb = self.embedding(padded_ids) * seq_mask.unsqueeze(-1)

        ns = batch["NS"]

        uid_emb = self._embed_sparse(ns["sparse_features"]["uid"])
        item_emb = self._embed_sparse(ns["sparse_features"]["item_id"])
        dense_enc = self.piecewise_encoder(ns["dense_features"])

        artist = ns["multivalent_features"]["artist_ids"]
        album = ns["multivalent_features"]["album_ids"]
        artist_emb = embed_multivalent(self.embedding, artist["values"], artist["lengths"])
        album_emb = embed_multivalent(self.embedding, album["values"], album["lengths"])

        return {
            "seq_features": [seq_emb],
            "seq_masks": [seq_mask],
            "ns_groups": [dense_enc, uid_emb, item_emb, artist_emb, album_emb],
        }
