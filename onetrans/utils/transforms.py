import numpy as np
import torch
import polars as pl
from typing import List
from onetrans.ext.yambda.dataset import BinaryRankinSequentialDataset
from onetrans.run.config import CORE_MIN_INTERACTIONS_PER_ITEM

class Transform:
    def __call__(self, x):
        raise NotImplementedError


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


class IdMapper(Transform):
    def __call__(self, listens, id_columns : List[str] = ["item_id"]):
        for id_col in id_columns:
            iid_mapping = (
                listens
                .select(pl.col(id_col).unique().sort())
                .with_row_index(name="new_item_id")
            )
            listens = (
                listens
                .join(iid_mapping, on=id_col, how="inner")
                .drop(id_col)
                .rename({"new_item_id" : id_col})
            )
        return listens

class MultiIdMapper(Transform):
    def __call__(
        self,
        listens : pl.DataFrame,
        id_columns : List[str] = ["album_ids", "artist_ids"]
    ):
        for id_col in id_columns:
            id_mapping = (
                listens
                .select(pl.col(id_col).fill_null([0]).explode())
                .select(pl.col(id_col).unique().sort())
                .with_row_index(name="new_id")
            )
            remapped = (
                listens
                .with_row_index(name="row_id")
                .select(["row_id", pl.col(id_col).fill_null([0])])
                .explode(id_col)
                .join(id_mapping, how="left", on=id_col, maintain_order="left")
                .group_by("row_id", maintain_order=True)
                .agg(pl.col("new_id").alias(id_col))
            )
            listens = (
                listens
                .with_row_index(name="row_id")
                .drop(id_col)
                .join(remapped, how="left", on="row_id", maintain_order="left")
                .drop("row_id")
            )
        return listens




class CoreFiltrator(Transform):
    def __init__(self, min_freq=CORE_MIN_INTERACTIONS_PER_ITEM):
        self.min_freq = min_freq
    def __call__(self, listens):
        relevant_items = (
            listens
            .group_by(pl.col("item_id"))
            .len(name="occurences")
            .filter(pl.col("occurences") >= self.min_freq)
            .select("item_id")
        )
        listens = (
            listens
            .join(relevant_items, how="semi", on="item_id")
        )
        return listens
