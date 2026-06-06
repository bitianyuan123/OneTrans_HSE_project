import polars as pl
from typing import Any, Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List
from onetrans.run.config import DENSE_COLUMNS
import numpy as np

class BinaryRankinArchive:
    def __init__(
        self,
        listens
    ):
        # Sort once so every per-user structure shares the same row order.
        s = listens.sort(["uid", "timestamp"])

        # Single group_by pass to build the full per-user history lists.
        agg = s.group_by("uid", maintain_order=True).agg(
            pl.col("item_id"),
            pl.col("is_like"),
            pl.col("is_full_play"),
            pl.col("timestamp"),
            pl.col("artist_ids"),
            pl.col("album_ids"),
        )
        uids = agg.get_column("uid").to_list()
        self.histories         = dict(zip(uids, agg.get_column("item_id").to_list()))
        self.target_likes      = dict(zip(uids, agg.get_column("is_like").to_list()))
        self.target_full_plays = dict(zip(uids, agg.get_column("is_full_play").to_list()))
        self.timestamps        = dict(zip(uids, agg.get_column("timestamp").to_list()))
        self.artist_ids        = dict(zip(uids, agg.get_column("artist_ids").to_list()))
        self.album_ids         = dict(zip(uids, agg.get_column("album_ids").to_list()))

        # Dense features are only ever read at a single position `t`, and only
        # for "complicated" pairs (where a label differs from its neighbour).
        # So we compute that mask vectorized and keep dense rows for those only.
        def prev(col: str) -> pl.Expr:
            return pl.col(col).shift(1).over("uid")

        def nxt(col: str) -> pl.Expr:
            return pl.col(col).shift(-1).over("uid")

        def complicated(col: str) -> pl.Expr:
            # differs from previous, or from next when a next row exists
            return (pl.col(col) != prev(col)) | (
                nxt(col).is_not_null() & (pl.col(col) != nxt(col))
            )

        interesting = (
            s.with_columns(pl.int_range(pl.len()).over("uid").alias("__t"))
            .filter(
                (pl.col("__t") >= 1)
                & (complicated("is_like") | complicated("is_full_play"))
            )
        )

        # Compact float32 matrix of dense features, one row per interesting pair.
        self.dense_matrix = (
            interesting.select(list(DENSE_COLUMNS))
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        pair_uids = interesting.get_column("uid").to_list()
        pair_ts = interesting.get_column("__t").to_list()
        pair_timestamps = interesting.get_column("timestamp").to_list()
        self.dense_index = {
            (uid, t): i for i, (uid, t) in enumerate(zip(pair_uids, pair_ts))
        }
        # (uid, t, timestamp) of every complicated pair; the Dataset filters
        # these by timestamp into train / test instead of rescanning history.
        self.interesting_pairs = list(zip(pair_uids, pair_ts, pair_timestamps))


class BinaryRankinSequentialDataset(Dataset):
    def __init__(
      self,
      archive : BinaryRankinArchive,
      max_seq_len: int = 100,
      is_train : bool = True,
      timestamp_test_start : int = 67
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.archive = archive

        for uid, t, ts in archive.interesting_pairs:
            if self.is_train and ts < timestamp_test_start:
                self.sequences.append((uid, t))
            elif not self.is_train and ts >= timestamp_test_start:
                self.sequences.append((uid, t))


    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid, t = self.sequences[idx]
        start = max(0, t - self.max_seq_len)
        out = {
            "S": {
                "history" : {
                    "item_id" : self.archive.histories[uid][start:t],
                    "album_id" : self.archive.album_ids[uid][start:t],
                    "artist_id" : self.archive.artist_ids[uid][start:t],
                    "length" : t - start
                },
            },
            "NS" : {
                "uid" : uid,
                "timestamp" : self.archive.timestamps[uid][t],
                "dense_features" : self.archive.dense_matrix[self.archive.dense_index[(uid, t)]],
                "multivalent_features" : {
                    "artist_id" : self.archive.artist_ids[uid][t],
                    "album_id" : self.archive.album_ids[uid][t],
                }
            },
            "targets" : {
                "is_like" : self.archive.target_likes[uid][t],
                "is_full_play": self.archive.target_full_plays[uid][t],
            }
        }
        return out
