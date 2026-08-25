"""一致性哈希路由（P1：user → owner worker 数据本地化）。

把 ``user_id`` 映射到 owner shard/worker，保证同一用户的 KV 与其 owner 共存，
读写本地命中；节点增减时只需迁移 ``O(k/n)`` 比例的键。

提供两种算法：

- :class:`JumpConsistentHash`：Lamping-Veach 2014「跳变哈希」，O(ln n) 无状态，
  桶数固定（分片数已知）时的首选，remap 最优。
- :class:`RingHash`：虚拟节点环（ketama 风格），支持任意节点集合动态增删。

键哈希统一用 sha256（稳定、跨进程一致），**不使用** Python 内置 ``hash()``
（其带进程随机盐，跨进程/重启不可复现）。

对应设计文档 §3.6「按 user_id 一致性哈希路由到 owner worker」与 §5 元数据/版本失效。
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Any, Iterable

_SHA256 = hashlib.sha256


def hash64(key: str) -> int:
    """稳定 64 位键哈希（sha256 前 8 字节，大端）。"""
    return int.from_bytes(_SHA256(key.encode("utf-8")).digest()[:8], "big")


class JumpConsistentHash:
    """跳变一致性哈希：固定桶数下的最小 remap 路由。

    :param num_shards: 分片/worker 数量（≥1）。
    """

    def __init__(self, num_shards: int) -> None:
        if num_shards <= 0:
            raise ValueError("num_shards 必须 ≥ 1")
        self.num_shards = num_shards

    def shard_of(self, key: str) -> int:
        """返回 ``key`` 归属的分片索引 ``[0, num_shards)``。"""
        k = hash64(key)
        b = -1
        j = 0
        while j < self.num_shards:
            b = j
            k = (k * 2862933555777941757 + 1) & 0xFFFFFFFFFFFFFFFF
            j = int((b + 1) * float(1 << 31) / float((k >> 33) + 1))
        return b

    def __call__(self, key: str) -> int:
        return self.shard_of(key)


class RingHash:
    """虚拟节点一致性哈希环，支持动态增删节点。"""

    def __init__(self, vnodes_per_node: int = 128) -> None:
        self.vnodes_per_node = vnodes_per_node
        self._tokens: list[int] = []  # 升序虚拟节点 token
        self._owner: list[Any] = []  # 与 _tokens 一一对应的 node
        self._nodes: set[Any] = set()

    @property
    def nodes(self) -> set[Any]:
        return set(self._nodes)

    def add_node(self, node: Any) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.vnodes_per_node):
            token = hash64(f"{node}#{i}")
            idx = bisect.bisect_left(self._tokens, token)
            self._tokens.insert(idx, token)
            self._owner.insert(idx, node)

    def remove_node(self, node: Any) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        keep_t: list[int] = []
        keep_o: list[Any] = []
        for t, o in zip(self._tokens, self._owner):
            if o != node:
                keep_t.append(t)
                keep_o.append(o)
        self._tokens = keep_t
        self._owner = keep_o

    def shard_of(self, key: str) -> Any:
        if not self._tokens:
            raise RuntimeError("环为空，请先 add_node")
        h = hash64(key)
        idx = bisect.bisect_right(self._tokens, h)
        if idx == len(self._tokens):
            idx = 0
        return self._owner[idx]


class Router:
    """路由门面：把 ``user_id`` 映射到 owner shard 索引，内部用一致性哈希。"""

    def __init__(self, num_shards: int | None = None, ring: RingHash | None = None) -> None:
        if ring is not None:
            self._ring: RingHash | None = ring
            self._jump: JumpConsistentHash | None = None
        else:
            if num_shards is None or num_shards <= 0:
                raise ValueError("Router 需要 num_shards 或 ring")
            self._ring = None
            self._jump = JumpConsistentHash(num_shards)

    def route(self, user_id: str) -> int:
        """返回 user_id 的 owner shard 索引。"""
        if self._jump is not None:
            return self._jump.shard_of(user_id)
        assert self._ring is not None
        return int(self._ring.shard_of(user_id))


def remap_ratio(old: Iterable[int], new: Iterable[int]) -> float:
    """两套分片下键变更的比例（用于测试 / 容量评估）。"""
    olds = list(old)
    news = list(new)
    assert len(olds) == len(news), "两套映射必须等长"
    if not olds:
        return 0.0
    return sum(1 for a, b in zip(olds, news) if a != b) / len(olds)