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
        self.histories = dict(
            listens.group_by("uid")
            .agg(pl.col("item_id").sort_by("timestamp"))
            .iter_rows()
        )
        self.target_likes = dict(
            listens.group_by("uid")
            .agg(pl.col("is_like").sort_by("timestamp"))
            .iter_rows()
        )
        self.target_full_plays = dict(
            listens.group_by("uid")
            .agg(pl.col("is_full_play").sort_by("timestamp"))
            .iter_rows()
        )
        self.timestamps = dict(
            listens.group_by("uid")
            .agg(pl.col("timestamp").sort_by("timestamp"))
            .iter_rows()
        )
        self.artist_ids = dict(
            listens.group_by("uid")
            .agg(pl.col("artist_ids").sort_by("timestamp"))
            .iter_rows()
        )
        self.album_ids = dict(
            listens.group_by("uid")
            .agg(pl.col("album_ids").sort_by("timestamp"))
            .iter_rows()
        )

        df_agg = (
            listens
            .sort("timestamp")
            .group_by("uid")
            .agg(
                pl.concat_list(DENSE_COLUMNS).alias("dense_matrix")
            )
        )
        self.dense = {
            uid: np.array(matrix) 
            for uid, matrix in df_agg.iter_rows()
        }


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

        for uid, history in archive.histories.items():
            for t in range(1, len(history)):
                complicated_case_like = archive.target_likes[uid][t] != archive.target_likes[uid][t - 1]
                complicated_case_full_play = archive.target_full_plays[uid][t] != archive.target_full_plays[uid][t - 1]
                if t + 1 < len(history):
                    complicated_case_like |= archive.target_likes[uid][t] != archive.target_likes[uid][t + 1]
                    complicated_case_full_play |= archive.target_full_plays[uid][t] != archive.target_full_plays[uid][t + 1]
                complicated_case = complicated_case_like or complicated_case_full_play
                if not complicated_case:
                    continue
                if self.is_train and archive.timestamps[uid][t] < timestamp_test_start:
                    self.sequences.append((uid, t))
                elif not self.is_train and archive.timestamps[uid][t] >= timestamp_test_start:
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
                "dense_features" : self.archive.dense[uid][t],
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
