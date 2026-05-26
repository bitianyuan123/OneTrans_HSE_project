from __future__ import annotations

from typing import Callable, Optional, Sequence, Union, Any

import polars as pl
from torch.utils.data import Dataset as TorchDataset, Subset


class Dataset(TorchDataset):
    def __init__(
        self,
        dataset: list[dict[str, Any]] | pl.DataFrame | TorchDataset,
        transform: Optional[Union[Callable, Sequence[Callable]]] = None,
    ) -> dict[str, Any]:
        super().__init__()
        self.dataset = dataset
        if transform is None:
            self._transforms = []
        elif callable(transform):
            self._transforms = [transform]
        else:
            self._transforms = list(transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if isinstance(self.dataset, pl.DataFrame):
            sample = self.dataset.row(index, named=True)
        else:
            sample = self.dataset[index]
        for t in self._transforms:
            sample = t(sample)
        return sample

    def take(self, n: int) -> Dataset:
        if isinstance(self.dataset, pl.DataFrame):
            new_data = self.dataset.head(n)
        elif isinstance(self.dataset, list):
            new_data = self.dataset[:n]
        else:
            new_data = Subset(self.dataset, range(min(n, len(self))))
        return Dataset(
            new_data,
            transform=self._transforms if self._transforms else None,
        )