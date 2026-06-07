from typing import Any
import mmh3
import torch
from torch import nn
from torch.nn.functional import embedding_bag


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


class MultihashEmbedding(nn.Module):
    _PRIMES = [2654435761, 2246822519, 3266489917, 668265263, 374761393,
               1640531527, 1000000007, 998244353, 479001599, 1234567891]

    def __init__(self, cardinality: int, embed_dim: int, num_hashes: int = 2):
        super().__init__()
        assert 1 <= num_hashes <= len(self._PRIMES), \
            f"num_hashes must be in [1, {len(self._PRIMES)}]"
        self.cardinality = cardinality
        self.num_hashes = num_hashes
        self.embedding = nn.Embedding(cardinality, embed_dim)

    def _hash(self, ids: torch.Tensor, prime: int) -> torch.Tensor:
        return (ids.long() * prime) % self.cardinality

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (*shape,)  →  out: (*shape, embed_dim)
        buckets = torch.stack(
            [self._hash(ids, self._PRIMES[h]) for h in range(self.num_hashes)],
            dim=-1,
        )  # (*shape, num_hashes)
        embeds = self.embedding(buckets)  # (*shape, num_hashes, embed_dim)
        return embeds.mean(dim=-2)        # (*shape, embed_dim)


class StandardMultivalentEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # values: (N_flat,)  lengths: (B,)  →  (B, embed_dim)
        B = lengths.shape[0]
        offsets = torch.zeros(B, dtype=torch.long, device=values.device)
        offsets[1:] = lengths.cumsum(dim=0)[:-1]
        return embedding_bag(
            values.contiguous(),
            self.embedding.weight,
            offsets=offsets,
            mode="mean",
            sparse=False,
        )


class MultihashMultivalentEmbedding(nn.Module):
    _PRIMES = [2654435761, 2246822519, 3266489917, 668265263, 374761393,
               1640531527, 1000000007, 998244353, 479001599, 1234567891]

    def __init__(self, cardinality: int, embed_dim: int, num_hashes: int = 2):
        super().__init__()
        assert 1 <= num_hashes <= len(self._PRIMES)
        self.cardinality = cardinality
        self.num_hashes = num_hashes
        self.embedding = nn.Embedding(cardinality, embed_dim)

    def _hash(self, values: torch.Tensor, prime: int) -> torch.Tensor:
        return (values.long() * prime) % self.cardinality

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # values: (N_flat,)  lengths: (B,)  →  (B, embed_dim)
        B = lengths.shape[0]
        offsets = torch.zeros(B, dtype=torch.long, device=values.device)
        offsets[1:] = lengths.cumsum(dim=0)[:-1]
        outputs = []
        for h in range(self.num_hashes):
            hashed = self._hash(values, self._PRIMES[h])
            emb = embedding_bag(
                hashed.contiguous(),
                self.embedding.weight,
                offsets=offsets,
                mode="mean",
                sparse=False,
            )
            outputs.append(emb)
        return torch.stack(outputs, dim=0).mean(dim=0)  # (B, embed_dim)
