import polars as pl
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from typing import Literal
from onetrans.run.config import (
    # DENSE_COLUMNS, SPARSE_COLUMNS, MULTIVALENT_COLUMNS, LABEL_COLUMNS,
    TEST_INTERVAL_SECONDS, CORE_MIN_INTERACTIONS_PER_ITEM
)
from onetrans.data.transforms import (
    FeatureJoiner, 
    IdMapper,
    CoreFiltrator,
    MultiIdMapper
)
from onetrans.ext.yambda.dataset import BinaryRankinSequentialDataset, BinaryRankinArchive
from onetrans.utils import collate_fn
from onetrans.nn.encoders.multihash import MultihashTransform
from typing import Tuple
from datasets import Dataset, DatasetDict, load_dataset



# YambdaDataset wrapper class (https://huggingface.co/datasets/yandex/yambda)
class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]


class DataCookinYambdaRank:
    def __init__(self):
        self.purpose = "Prepare data for training Binary Ranker on Yambda-50M Dataset"

    def cook(self, dataset_config) -> Tuple[pl.DataFrame, pl.DataFrame]:
        '''
            Load && preprocess Yambda
        '''
        path = hf_hub_download(
            repo_id="matfu21/yambda-50m-lag-features",
            repo_type="dataset",
            filename="listens.parquet",
        )
        listens = pl.scan_parquet(path)

        if dataset_config.max_users is not None:
            all_users = listens.select("uid").unique().collect()
            n = min(dataset_config.max_users, len(all_users))
            listens = listens.join(all_users.sample(n=n, seed=42).lazy(), on="uid")

        filtrator = CoreFiltrator(min_freq=CORE_MIN_INTERACTIONS_PER_ITEM)
        listens = filtrator(listens).collect(engine="streaming").lazy()

        yambda_dataset = YambdaDataset(
            dataset_type=dataset_config.dataset_type,
            dataset_size=dataset_config.dataset_size
        )
        albums = yambda_dataset.album_item_mapping().to_polars().lazy()
        artists = yambda_dataset.artist_item_mapping().to_polars().lazy()

        joiner = FeatureJoiner()
        listens = joiner(listens, artists, albums)

        iid_mapper = IdMapper()
        listens = iid_mapper(listens, id_columns=["item_id", "uid"])
        ids_mapper = MultiIdMapper()
        listens = ids_mapper(listens, id_columns=["album_ids", "artist_ids"])

        listens = listens.collect()

        timestamp_test_start = listens.select(pl.col("timestamp").max()).item() - TEST_INTERVAL_SECONDS
        return listens, timestamp_test_start

    def run(self, data_config):
        '''
            MLil znatno nash MLin
        '''
        listens, timestamp_test_start = self.cook(data_config)
        archive = BinaryRankinArchive(listens)
        train_set, test_set = (
            BinaryRankinSequentialDataset(archive, timestamp_test_start=timestamp_test_start),
            BinaryRankinSequentialDataset(archive, is_train=False, timestamp_test_start=timestamp_test_start)
        )

        train_loader = DataLoader(
            train_set, 
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_set,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            collate_fn=collate_fn
        )
        return train_loader, test_loader
