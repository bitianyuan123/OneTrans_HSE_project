import polars as pl
import torch.nn as nn

from onetrans.ext.yambda.embedder import YambdaEmbedder
from onetrans.nn.encoders.multihash import (
    MultihashEmbedding,
    MultihashMultivalentEmbedding,
    StandardMultivalentEmbedding,
)
from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder
from onetrans.nn.tokenizer import STokenizer, NSGroupWiseTokenizer, NSAutoSplitTokenizer, OneTransTokenizer
from onetrans.baselines.one_trans import OneTrans
from onetrans.run.config import DENSE_COLUMNS


def build_model(
    archive,
    d_model,
    n_layers,
    n_heads,
    max_seq_len,
    device,
    merge="timestamp_agnostic",
    use_multihash=False,
    hash_cardinality=65536,
    num_hashes=2,
    ns_tokenizer="groupwise",
    l_ns=5,
):
    if use_multihash:
        item_embedding = MultihashEmbedding(hash_cardinality, d_model, num_hashes)
        artist_embedding = MultihashMultivalentEmbedding(hash_cardinality, d_model, num_hashes)
        album_embedding = MultihashMultivalentEmbedding(hash_cardinality, d_model, num_hashes)
    else:
        item_embedding = nn.Embedding(archive.meta["num_items"] + 1, d_model, padding_idx=0)
        artist_embedding = StandardMultivalentEmbedding(
            nn.Embedding(archive.meta["num_artists"] + 1, d_model)
        )
        album_embedding = StandardMultivalentEmbedding(
            nn.Embedding(archive.meta["num_albums"] + 1, d_model)
        )

    embedder = YambdaEmbedder(
        item_embedding=item_embedding,
        user_embedding=nn.Embedding(archive.meta["num_users"] + 1, d_model),
        artist_embedding=artist_embedding,
        album_embedding=album_embedding,
        piecewise_encoder=PiecewiseLinearEncoder.from_dataset(
            pl.from_numpy(archive.dense_matrix, schema=list(DENSE_COLUMNS))
        ),
        max_seq_len=max_seq_len,
    )

    s_tok = STokenizer(d_model, in_dims=[embedder.seq_in_dim], merge=merge)
    if ns_tokenizer == "autosplit":
        ns_tok = NSAutoSplitTokenizer(d_model, l_ns=l_ns, in_dims=embedder.ns_group_dims)
    else:
        ns_tok = NSGroupWiseTokenizer(d_model, in_dims=embedder.ns_group_dims)
    tokenizer = OneTransTokenizer(s_tok, ns_tok, d_model, max_seq_len)

    backbone = OneTrans(
        d_model=d_model,
        num_blocks=n_layers,
        num_heads=n_heads,
        max_seq_len=max_seq_len,
        ns_tokens_num=ns_tok.n_ns_tokens,
    )

    return embedder.to(device), tokenizer.to(device), backbone.to(device)
