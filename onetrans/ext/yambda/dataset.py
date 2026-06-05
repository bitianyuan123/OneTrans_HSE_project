import polars as pl
from typing import Any, Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RankerDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        transforms: list[Callable[[Any], Any]],
        label_columns: list[str],
        dense_columns: list[str],
        sparse_columns: list[str],
        multivalent_columns: list[str],
        batch_size: int,
    ):
        self.batches = []
        for i in tqdm(range((len(df) + batch_size - 1) // batch_size)):
            batch_df = df[i * batch_size : min((i + 1) * batch_size, len(df))]
            yet_another_batch = {
                "labels" : {col : batch_df[col].to_torch() for col in label_columns},
                "dense_features" :  batch_df.select(dense_columns).to_torch().to(torch.float32),
                "sparse_features" : {col : batch_df[col].to_torch() for col in sparse_columns},
                "multivalent_features" : {
                    col : {
                        "values" : batch_df.select(pl.col(col).fill_null([])).explode(col).to_torch().flatten(),
                        "lengths" : batch_df.select(pl.col(col).list.len().fill_null(0)).to_torch().flatten()
                    } for col in multivalent_columns
                },
                "meta" : {
                    "timestamp" : batch_df["timestamp"].to_torch(),
                    "uid" : batch_df["uid"].to_torch(),
                    "item_id" : batch_df["item_id"].to_torch()
                }
            }
            for f in transforms:
                yet_another_batch = f(yet_another_batch)
            self.batches.append(yet_another_batch)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.batches[idx]