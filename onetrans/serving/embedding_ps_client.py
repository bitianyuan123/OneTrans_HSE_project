"""独立稀疏参数服务器（PS）数据面客户端 + 本地分片参照实现（fabric ①）。

稀疏 embedding 独立服务化（C++ brpc 侧见 ``deploy/ps/``），本模块是 Python 侧：

- :class:`ShardedEmbeddingTable`：按 id 稳定哈希分片的 embeding 表（每分片独立锁），
  与 C++ 侧 ``ShardedEmbeddingTable`` 语义一致（分片路由统一 Knuth 乘法哈希，G10）；
- :class:`LocalEmbeddingPS`：进程内多表 PS（供单机数值/并发基准）；
- :class:`EmbeddingPSClient`：独立 PS 数据面客户端，``local`` 走进程内分片表，
  ``remote`` 走 brpc RPC（wire 契约见 ``deploy/ps/embedding_service.proto``）。

未命中（miss）语义：返回 0 向量并由上层按「seed 兜底哈希嵌入」重建（对应权重版本化
加载的最差路径），或由客户端显式 ``seed_fallback`` 直接生成确定性兜底嵌入。
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import torch
from torch import Tensor

# Knuth 乘法哈希常数：与 deploy/ps/embedding_server.cc 的 detail::ShardOf 同一标准（G10）。
# 读写两侧必须对同一 id 落同一分片，故 Python 侧不再用 router.hash64（sha256）。
_KNUTH = 0x9E3779B97F4A7C15


class ShardedEmbeddingTable:
    """按 id 稳定哈希分片的内存嵌入表（每分片细粒度锁，无全局锁）。"""

    def __init__(self, num_shards: int, dim: int, seed: int = 0) -> None:
        if num_shards <= 0 or dim <= 0:
            raise ValueError("num_shards/dim 必须 > 0")
        self.num_shards = num_shards
        self.dim = dim
        self.seed = seed
        self._shards: list[dict[int, list[float]]] = [{} for _ in range(num_shards)]
        self._locks: list[threading.Lock] = [threading.Lock() for _ in range(num_shards)]
        self._version = 0

    # -- 路由 ------------------------------------------------------------ #
    def shard_of(self, feat_id: int) -> int:
        # 与 C++ detail::ShardOf 逐位对齐：
        #   1) 负 id 按二补码回绕到 uint64（对齐 static_cast<uint64_t>）；
        #   2) 乘法结果按 2^64 截断（对齐 uint64 自然溢出）；
        #   3) 再对 num_shards 取模。
        return (((feat_id & 0xFFFFFFFFFFFFFFFF) * _KNUTH) & 0xFFFFFFFFFFFFFFFF) % self.num_shards

    # -- 读写 ------------------------------------------------------------ #
    def set(self, feat_id: int, weights: list[float]) -> None:
        if len(weights) != self.dim:
            raise ValueError(f"向量维度须为 {self.dim}")
        s = self.shard_of(feat_id)
        with self._locks[s]:
            self._shards[s][feat_id] = list(weights)
        self.bump_version()  # 与 C++ 侧 Set() 每写触发版本递增一致

    def set_many(self, ids: Tensor, weights: Tensor) -> None:
        """批量写入：weights ``[N, dim]`` 与 ``ids`` 对齐（逐 id 递增版本）。"""
        wlist = weights.detach().to("cpu").tolist()
        for i, fid in enumerate(ids.tolist()):
            self.set(int(fid), wlist[i])

    def get(self, feat_id: int) -> Optional[list[float]]:
        s = self.shard_of(feat_id)
        with self._locks[s]:
            return self._shards[s].get(feat_id)

    def lookup(self, ids: Tensor, *, seed_fallback: bool = True) -> Tensor:
        """批量查表，返回 ``[N, dim]``；miss 时按 seed 兜底哈希嵌入。"""
        out = torch.empty(len(ids), self.dim, dtype=torch.float32)
        for i, fid in enumerate(ids.tolist()):
            w = self.get(int(fid))
            if w is None:
                out[i] = self._seed_embedding(int(fid)) if seed_fallback else 0.0
            else:
                out[i] = torch.tensor(w, dtype=torch.float32)
        return out

    def bump_version(self) -> int:
        self._version += 1
        return self._version

    @property
    def version(self) -> int:
        return self._version

    def size(self) -> int:
        return sum(len(s) for s in self._shards)

    def _seed_embedding(self, feat_id: int) -> Tensor:
        """确定性种子兜底：由 ``(feat_id, seed)`` 生成稳定伪随机向量（权重版本化最差路径）。"""
        gen = torch.Generator().manual_seed((feat_id * 0x9E3779B97F4A7C15 + self.seed) & 0xFFFFFFFF)
        return torch.randn(self.dim, generator=gen)


class LocalEmbeddingPS:
    """进程内多表 PS（表名 → :class:`ShardedEmbeddingTable`）。"""

    def __init__(self, num_shards: int = 64, dim: int = 64, seed: int = 0) -> None:
        self.num_shards = num_shards
        self.dim = dim
        self.seed = seed
        self._tables: dict[str, ShardedEmbeddingTable] = {}
        self._lock = threading.Lock()

    def table(self, name: str) -> ShardedEmbeddingTable:
        with self._lock:
            if name not in self._tables:
                self._tables[name] = ShardedEmbeddingTable(self.num_shards, self.dim, self.seed)
            return self._tables[name]

    def set_many(self, table: str, ids: Tensor, weights: Tensor) -> None:
        self.table(table).set_many(ids, weights)

    def lookup(self, table: str, ids: Tensor, *, seed_fallback: bool = True) -> Tensor:
        return self.table(table).lookup(ids, seed_fallback=seed_fallback)

    def version(self, table: str) -> int:
        return self.table(table).version


class EmbeddingPSClient:
    """独立 PS 数据面客户端（fabric ①）。

    推荐路径：``local`` 模式走进程内分片表（数值/并发基准）；``remote`` 模式发 brpc
    RPC（table/dim/version wire 契约见 ``deploy/ps/embedding_service.proto``）。
    """

    def __init__(
        self,
        table: str,
        dim: int,
        *,
        local: LocalEmbeddingPS | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.table = table
        self.dim = dim
        self._local = local
        self.host = host
        self.port = port

    @classmethod
    def local(cls, table: str, dim: int, num_shards: int = 64, seed: int = 0) -> "EmbeddingPSClient":
        return cls(table, dim, local=LocalEmbeddingPS(num_shards, dim, seed))

    def lookup(self, ids: Tensor) -> Tensor:
        if self._local is not None:
            return self._local.lookup(self.table, ids)
        # remote：经 brpc 调 deploy/ps 的 EmbeddingService.Lookup（此处占位，定义 wire 契约）
        raise NotImplementedError(
            f"remote PS 未接入：请用 brpc 客户端调 {self.host}:{self.port} 的 EmbeddingService.Lookup"
        )

    def version(self) -> int:
        if self._local is not None:
            return self._local.version(self.table)
        raise NotImplementedError("remote PS 未接入")