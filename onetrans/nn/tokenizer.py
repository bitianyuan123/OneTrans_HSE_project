import torch
from torch import nn
from torch.nn.functional import embedding_bag

from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder


def _mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=False),
    )


def _embed_multivalent(embedding, values, lengths):
    batch_size = lengths.shape[0]
    if values.dim() == 1:
        values = values.unsqueeze(1)
    num_hashes = values.shape[1]
    offsets = torch.zeros(batch_size, dtype=torch.long, device=values.device)
    offsets[1:] = lengths.cumsum(dim=0)[:-1]
    outputs = []
    for h in range(num_hashes):
        emb = embedding_bag(
            embedding.weight,
            values[:, h].contiguous(),
            offsets=offsets,
            mode="mean",
            sparse=False,
        )
        outputs.append(emb)
    return torch.cat(outputs, dim=1)


class STokenizer(nn.Module):
    def __init__(self, d_model, num_items, item_embedding_dim, n_signals=3):
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, item_embedding_dim)
        self.proj = _mlp(item_embedding_dim + n_signals, d_model)

    def forward(self, seq_items, seq_signals, seq_mask):
        item_emb = self.item_embeddings(seq_items)
        event_emb = torch.cat([item_emb, seq_signals], dim=-1)
        tokens = self.proj(event_emb)
        tokens = tokens * seq_mask.unsqueeze(-1)
        return tokens, seq_mask


class NSGroupWiseTokenizer(nn.Module):
    N_NS_TOKENS = 5

    def __init__(self, d_model, embedding, piecewise_encoder, embedding_dim, num_hashes=1):
        super().__init__()
        self.piecewise_encoder = piecewise_encoder
        self.embedding = embedding
        emb_out = num_hashes * embedding_dim

        self.mlp_dense = _mlp(piecewise_encoder.out_features, d_model)
        self.mlp_uid = _mlp(embedding_dim, d_model)
        self.mlp_item = _mlp(embedding_dim, d_model)
        self.mlp_artist = _mlp(emb_out, d_model)
        self.mlp_album = _mlp(emb_out, d_model)

    @property
    def n_ns_tokens(self):
        return self.N_NS_TOKENS

    def forward(self, batch):
        dense_enc = self.piecewise_encoder(batch["dense"])
        uid_emb = self.embedding(batch["sparse"][:, 0])
        item_emb = self.embedding(batch["sparse"][:, 1])

        artist = batch["multivalent"]["artist_ids"]
        album = batch["multivalent"]["album_ids"]
        artist_emb = _embed_multivalent(self.embedding, artist["values"], artist["lengths"])
        album_emb = _embed_multivalent(self.embedding, album["values"], album["lengths"])

        return torch.stack([
            self.mlp_dense(dense_enc),
            self.mlp_uid(uid_emb),
            self.mlp_item(item_emb),
            self.mlp_artist(artist_emb),
            self.mlp_album(album_emb),
        ], dim=1)


class NSAutoSplitTokenizer(nn.Module):
    def __init__(self, d_model, l_ns, embedding, piecewise_encoder, embedding_dim, num_hashes=1):
        super().__init__()
        self.piecewise_encoder = piecewise_encoder
        self.embedding = embedding
        self._l_ns = l_ns
        self._d_model = d_model

        in_dim = (
            piecewise_encoder.out_features
            + embedding_dim * 2
            + num_hashes * embedding_dim * 2
        )
        self.proj = _mlp(in_dim, d_model * l_ns)

    @property
    def n_ns_tokens(self):
        return self._l_ns

    def forward(self, batch):
        dense_enc = self.piecewise_encoder(batch["dense"])
        uid_emb = self.embedding(batch["sparse"][:, 0])
        item_emb = self.embedding(batch["sparse"][:, 1])

        artist = batch["multivalent"]["artist_ids"]
        album = batch["multivalent"]["album_ids"]
        artist_emb = _embed_multivalent(self.embedding, artist["values"], artist["lengths"])
        album_emb = _embed_multivalent(self.embedding, album["values"], album["lengths"])

        ns_concat = torch.cat([dense_enc, uid_emb, item_emb, artist_emb, album_emb], dim=1)
        projected = self.proj(ns_concat)
        return projected.reshape(projected.shape[0], self._l_ns, self._d_model)


class OneTransTokenizer(nn.Module):
    def __init__(self, s_tokenizer, ns_tokenizer, d_model):
        super().__init__()
        self.s_tokenizer = s_tokenizer
        self.ns_tokenizer = ns_tokenizer
        self.rms_norm = nn.RMSNorm(d_model)

    @property
    def n_ns_tokens(self):
        return self.ns_tokenizer.n_ns_tokens

    def forward(self, batch):
        s_tokens, s_mask = self.s_tokenizer(
            batch["seq_items"],
            batch["seq_signals"],
            batch["seq_mask"],
        )

        ns_tokens = self.ns_tokenizer(batch)
        B, L_NS, _ = ns_tokens.shape
        ns_mask = torch.ones(B, L_NS, dtype=torch.bool, device=ns_tokens.device)

        tokens = torch.cat([s_tokens, ns_tokens], dim=1)
        mask = torch.cat([s_mask, ns_mask], dim=1)

        return self.rms_norm(tokens), mask
