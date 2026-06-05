import polars as pl
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from typing import Literal
from onetrans.run.config import DENSE_COLUMNS, SPARSE_COLUMNS, MULTIVALENT_COLUMNS, LABEL_COLUMNS
from onetrans.data.transforms import PreferencePairsExtractor, FeatureJoiner, TemporalTranTestSplitter
from onetrans.ext.yambda.dataset import RankerDataset
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
        listens = pl.read_parquet(path)

        extractor = PreferencePairsExtractor()
        listens = extractor(listens)
    
        yambda_dataset = YambdaDataset(
            dataset_type=dataset_config.dataset_type,
            dataset_size=dataset_config.dataset_size
        )
        albums = yambda_dataset.album_item_mapping().to_polars()
        artists = yambda_dataset.artist_item_mapping().to_polars()

        joiner = FeatureJoiner()
        listens = joiner(listens, artists, albums)

        time_splitter = TemporalTranTestSplitter()
        train_listens, test_listens = time_splitter(listens, test_last_seconds=30 * 24 * 60 * 60)
        return train_listens, test_listens

    def run(self, dataset_config):
        '''
            Full pipeline from raw data to train & test loaders
            --> for Transformer-like models (OneTrans, RankMixer): masked sequences of fixed length
            --> otherwise: flattened sequences of variable length ???
        '''
        train_listens, test_listens = self.cook(dataset_config)
        multihash_transform = MultihashTransform(
            sparse_features_config={
                'item_id': [1, 2, 3, 4, 5],
                'uid': [6, 7, 8, 9, 10],
            },
            sparse_features_name="sparse_features",
            multivalent_features_config={
                'artist_ids': [11, 12, 13, 14, 15],
                'album_ids': [16, 17, 18, 19, 20],
            },
            multivalent_features_name="multivalent_features",
            cardinality=65379,
        )

        train_dataset = RankerDataset(
            train_listens,
            [multihash_transform],
            label_columns=list(LABEL_COLUMNS),
            dense_columns=list(DENSE_COLUMNS),
            sparse_columns=list(SPARSE_COLUMNS),
            multivalent_columns=list(MULTIVALENT_COLUMNS),
            batch_size=10000,
        )

        test_dataset = RankerDataset(
            test_listens,
            [multihash_transform],
            label_columns=list(LABEL_COLUMNS),
            dense_columns=list(DENSE_COLUMNS),
            sparse_columns=list(SPARSE_COLUMNS),
            multivalent_columns=list(MULTIVALENT_COLUMNS),
            batch_size=1024,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch[0],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch[0],
        )

        return train_loader, test_loader
