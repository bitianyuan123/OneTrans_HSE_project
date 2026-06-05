import numpy as np
import torch
import polars as pl

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class ToNumpy(Transform):
    """ Convert all lists to numpy """

    def __init__(self, dtype=np.int64):
        super().__init__()
        self._dtype = dtype

    def __call__(self, sample):
        res = {}
        for key, value in sample.items():
            if isinstance(value, dict):
                res[key] = self.__call__(value)
            elif isinstance(value, list):
                res[key] = np.array(value, dtype=self._dtype)
            else:
                res[key] = value
        return res


class ToTorch(Transform):
    """Convert all lists or numpy arrays in torch tensors."""

    def __call__(self, obj):
        if isinstance(obj, dict):
            return {key: self.__call__(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return torch.tensor(obj)
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        else:
            return obj


class ToDevice(Transform):
    """Move obj to device."""

    def __init__(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ):
        self._device = device
        self._non_blocking = non_blocking

    def __call__(self, obj):
        def _to_device_recursive(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self._device, non_blocking=self._non_blocking)
            if isinstance(obj, dict):
                return {k: _to_device_recursive(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_device_recursive(x) for x in obj)
            return obj
        return _to_device_recursive(obj)

class PreferencePairsExtractor(Transform):
    def extract_preference_pairs(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .with_row_index(name="row_id")
            .sort(by=["uid", "timestamp", "row_id"])
            .with_columns(
                pl.col("is_like").shift(1).over(partition_by="uid", order_by="timestamp").alias("prev_like"),
                pl.col("is_like").shift(-1).over(partition_by="uid", order_by="timestamp").alias("next_like"),
                pl.col("is_full_play").shift(1).over(partition_by="uid", order_by="timestamp").alias("prev_fullplay"),
                pl.col("is_full_play").shift(-1).over(partition_by="uid", order_by="timestamp").alias("next_fullplay")
            )
            .with_columns(
                pl.any_horizontal(
                    pl.col("prev_like") != pl.col("is_like"), 
                    pl.col("next_like") != pl.col("is_like"), 
                    pl.col("prev_fullplay") != pl.col("is_full_play"), 
                    pl.col("next_fullplay") != pl.col("is_full_play")
                ).alias("relevant")
            )
            .filter(pl.col("relevant"))
            .drop(["row_id", "relevant", "prev_like", "next_like", "prev_fullplay", "next_fullplay"])
    )

    def __call__(self, df : pl.DataFrame) -> pl.DataFrame:
        return self.extract_preference_pairs(df)


class FeatureJoiner(Transform):
    def join_item_artist_album(
        self,
        listens: pl.DataFrame,
        artists: pl.DataFrame,
        albums: pl.DataFrame,
    ) -> pl.DataFrame:
        albums = (
            albums
            .group_by("item_id")
            .agg(pl.col("album_id"))
        )
        artists = (
            artists
            .group_by("item_id")
            .agg(pl.col("artist_id"))
        )
        return (
            listens
            .join(artists, how="left", on="item_id", maintain_order="left")
            .join(albums, how="left", on="item_id", maintain_order="left")
            .rename({
                "album_id" : "album_ids",
                "artist_id" : "artist_ids"
            })
        )

    def __call__(
        self,
        listens: pl.DataFrame,
        artists: pl.DataFrame,
        albums: pl.DataFrame
    ) -> pl.DataFrame:
        return self.join_item_artist_album(listens, artists, albums)


class TemporalTranTestSplitter(Transform):
    def temporal_train_test_split(
        self,
        df: pl.DataFrame,
        test_last_seconds: float,
        time_column: str = "timestamp",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        SPLIT_TS = df.select(pl.col("timestamp").max()).item() - test_last_seconds + 1
        return (
            df.filter(pl.col("timestamp") < SPLIT_TS),
            df.filter(pl.col("timestamp") >= SPLIT_TS)
        )

    def __call__(
        self, 
        df, 
        test_last_seconds : float, 
        time_column : str = "timestamp"
    ):
        return self.temporal_train_test_split(df, test_last_seconds, time_column)