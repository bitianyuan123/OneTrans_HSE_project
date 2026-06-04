import polars as pl
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from typing import Literal
import numpy as np

from onetrans.run.config import DatasetConfig, DENSE_COLUMNS, SPARSE_COLUMNS, MULTIVALENT_COLUMNS


class RankingDataset:
    """
    Универсальный загрузчик данных для разных моделей:
    - 'catboost', 'dcn', 'hiformer' – только NS-фичи (lag + sparse + multivalent)
    - 'one_trans', 'rank_mixer' – NS-фичи + последовательности (S-токены)
    """
    def __init__(self, config: DatasetConfig):
        self.config = config
        # Загружаем датасет с lag‑фичами
        path = hf_hub_download(
            repo_id="matfu21/yambda-50m-lag-features",
            repo_type="dataset",
            filename="listens.parquet",
        )
        df = pl.read_parquet(path)

        # Добавляем мультивалентные колонки (artist_ids, album_ids)
        # Для этого нужны маппинги из YambdaDataset, но для краткости опустим
        # (предполагаем, что они уже есть или будут добавлены отдельно)

        # Сортируем и строим окна истории для OneTrans/RankMixer
        self.df = df.sort(["uid", "timestamp"])

    def _build_history(self, seq_len: int):
        """Добавляет в self.df колонки истории длины seq_len."""
        history_cols = []
        for k in range(1, seq_len + 1):
            history_cols.append(
                pl.col("item_id").shift(k).over("uid").alias(f"seq_item_{k}")
            )
            for sig in ["is_like", "is_full_play", "is_skip"]:
                history_cols.append(
                    pl.col(sig).shift(k).over("uid").alias(f"seq_{sig}_{k}")
                )
        self.df = self.df.with_columns(history_cols)
        # Маска: реальный токен если item_id != 0
        self.seq_mask_cols = [f"seq_item_{k}" for k in range(1, seq_len + 1)]

    def get_dataset(
        self,
        model_name: Literal['catboost', 'dcn', 'hiformer', 'one_trans', 'rank_mixer'],
        split: Literal['train', 'test'] = 'train',
        seq_len: int = 20,
        target: str = 'is_full_play',
        test_last_days: int = 30,
    ) -> Dataset:
        """
        Возвращает torch Dataset для указанной модели.
        """
        # Разделение по времени
        max_ts = self.df["timestamp"].max()
        threshold = max_ts - test_last_days * 24 * 3600
        if split == 'train':
            df_split = self.df.filter(pl.col("timestamp") < threshold)
        else:
            df_split = self.df.filter(pl.col("timestamp") >= threshold)

        # Для моделей с последовательностью строим историю
        use_seq = model_name in ('one_trans', 'rank_mixer')
        if use_seq:
            self.seq_len = seq_len
            self._build_history(seq_len)
            # Пересоздаём df_split после добавления колонок
            if split == 'train':
                df_split = self.df.filter(pl.col("timestamp") < threshold)
            else:
                df_split = self.df.filter(pl.col("timestamp") >= threshold)

        # Создаём датасет
        return _ModelSpecificDataset(
            df=df_split,
            model_name=model_name,
            use_seq=use_seq,
            seq_len=seq_len if use_seq else 0,
            target=target,
        )


class _ModelSpecificDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        model_name: str,
        use_seq: bool,
        seq_len: int,
        target: str,
    ):
        self.df = df
        self.model_name = model_name
        self.use_seq = use_seq
        self.seq_len = seq_len
        self.target = target

        # Подготавливаем NS-часть (всегда)
        self.ns_dense = torch.tensor(
            df.select(DENSE_COLUMNS).fill_null(0).to_numpy(),
            dtype=torch.float32
        )
        self.ns_sparse = torch.tensor(
            df.select(SPARSE_COLUMNS).to_numpy(),
            dtype=torch.long
        )
        # Мультивалентные признаки
        self.multivalent = {}
        for col in MULTIVALENT_COLUMNS:
            if col in df.columns:
                lists = df[col].to_list()
                lengths = [len(lst) for lst in lists]
                values = [v for lst in lists for v in lst]
                self.multivalent[col] = {
                    'values': torch.tensor(values, dtype=torch.long) if values else torch.empty(0, dtype=torch.long),
                    'lengths': torch.tensor(lengths, dtype=torch.long),
                }
            else:
                # Пустые
                self.multivalent[col] = {
                    'values': torch.empty(0, dtype=torch.long),
                    'lengths': torch.zeros(len(df), dtype=torch.long),
                }

        # Последовательная часть (если нужна)
        if use_seq:
            seq_item_cols = [f"seq_item_{k}" for k in range(1, seq_len + 1)]
            self.seq_items = torch.tensor(
                df.select(seq_item_cols).fill_null(0).to_numpy(),
                dtype=torch.long
            )
            # Сигналы: is_like, is_full_play, is_skip
            signal_arrays = []
            for sig in ["is_like", "is_full_play", "is_skip"]:
                cols = [f"seq_{sig}_{k}" for k in range(1, seq_len + 1)]
                arr = df.select(cols).fill_null(0).to_numpy()
                signal_arrays.append(arr)
            self.seq_signals = torch.tensor(
                np.stack(signal_arrays, axis=-1),
                dtype=torch.float32
            )
            self.seq_mask = (self.seq_items != 0)

        # Целевая переменная (один таргет для простоты, можно расширить)
        self.labels = torch.tensor(
            df[target].fill_null(0).cast(pl.Float32).to_numpy(),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'dense': self.ns_dense[idx],
            'sparse': self.ns_sparse[idx],
            'multivalent': {k: {sub: v[sub][idx] if isinstance(v[sub], torch.Tensor) and v[sub].numel() > 0 else v[sub]
                                 for sub in v}
                            for k, v in self.multivalent.items()},
            'label': self.labels[idx],
        }
        # Для моделей с последовательностью добавляем seq-поля
        if self.use_seq:
            sample['seq_items'] = self.seq_items[idx]
            sample['seq_signals'] = self.seq_signals[idx]
            sample['seq_mask'] = self.seq_mask[idx]
        return sample