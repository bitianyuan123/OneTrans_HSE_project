from typing import Any
import mmh3
import torch


class MultihashTransform:
    def __init__(
            self,
            sparse_features_config: dict,
            sparse_features_name: str,
            multivalent_features_config: dict,
            multivalent_features_name: str,
            cardinality: int,
    ):
        self.sparse_features_config = sparse_features_config
        self.sparse_features_name = sparse_features_name
        self.multivalent_features_config = multivalent_features_config
        self.multivalent_features_name = multivalent_features_name
        self.cardinality = cardinality

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sparse_dict = sample[self.sparse_features_name]

        for feature_name, seeds in self.sparse_features_config.items():
            tensor = sparse_dict[feature_name]
            hashes = []

            for seed in seeds:
                hashed = torch.tensor(
                    [mmh3.hash(str(x.item()), seed=seed) for x in tensor],
                    dtype=torch.long
                )
                hashed = hashed % self.cardinality
                hashed = hashed % self.cardinality
                hashes.append(hashed)

            sparse_dict[feature_name] = torch.stack(hashes, dim=1)

        multi_dict = sample[self.multivalent_features_name]

        for feature_name, seeds in self.multivalent_features_config.items():
            values = multi_dict[feature_name]["values"]
            lengths = multi_dict[feature_name]["lengths"]

            hashes = []
            for seed in seeds:
                hashed = torch.tensor(
                    [mmh3.hash(str(x.item()), seed=seed) for x in values],
                    dtype=torch.long
                )
                hashed = hashed % self.cardinality
                hashed = hashed % self.cardinality
                hashes.append(hashed)

            multi_dict[feature_name]["values"] = torch.stack(hashes, dim=1)

        return sample
