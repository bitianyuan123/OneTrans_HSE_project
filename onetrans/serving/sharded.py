"""一致性哈希分片 KVStore：不同 user 路由到不同 owner shard（数据本地化）。

把 :class:`KVStore` 协议委托到 N 个后端 shard，按 ``user_id`` 经一致性哈希路由，
保证同一用户的 KV 与 owner worker/shards 共存（读写本地命中）。上层（worker）仍只依赖
:class:`KVStore` 接口，不感知分片。

对应设计文档 §3.6（按 user 一致性哈希路由）与 §5（KV 与 owner 同节点共存）。
"""

from __future__ import annotations

from typing import Any, Optional

from onetrans.serving.kv_store import (
    AppendResult,
    DeleteResult,
    DeltaKV,
    KVConfig,
    KVKey,
    KVStore,
    PutResult,
    UserKVRecord,
    build_kv_store,
)
from onetrans.serving.router import JumpConsistentHash, Router


class ShardedKVStore(KVStore):
    """按 user_id 一致性哈希分片的 KV 门面。"""

    def __init__(self, stores: list[KVStore], router: Router | None = None) -> None:
        if not stores:
            raise ValueError("stores 不能为空")
        self.stores = list(stores)
        self.router = router or Router(num_shards=len(self.stores))

    # -- 内部：路由 -------------------------------------------------------- #
    def shard_of(self, user_id: str) -> int:
        return self.router.route(user_id)

    def _store(self, key: KVKey) -> KVStore:
        return self.stores[self.shard_of(key.user_id)]

    # -- 生命周期 ---------------------------------------------------------- #
    def connect(self, conf: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        for s in self.stores:
            s.connect(None)

    def close(self) -> None:
        for s in self.stores:
            s.close()

    # -- 单对象读写 -------------------------------------------------------- #
    def put(self, rec: UserKVRecord) -> PutResult:
        return self._store(rec.key).put(rec)

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> Optional[UserKVRecord]:
        return self._store(key).get(key, layers=layers)

    def mget(self, keys: list[KVKey], *, layers: list[int] | None = None) -> list[Optional[UserKVRecord]]:
        groups: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(self.shard_of(k.user_id), []).append(i)
        out: list[Optional[UserKVRecord]] = [None] * len(keys)
        for shard_idx, idxs in groups.items():
            sub_keys = [keys[i] for i in idxs]
            for i, r in zip(idxs, self.stores[shard_idx].mget(sub_keys, layers=layers)):
                out[i] = r
        return out

    # -- 增量 append ------------------------------------------------------- #
    def append(self, delta: DeltaKV) -> AppendResult:
        return self._store(delta.key).append(delta)

    # -- 生命周期/失效 ----------------------------------------------------- #
    def delete(self, keys: list[KVKey]) -> DeleteResult:
        total = 0
        groups: dict[int, list[KVKey]] = {}
        for k in keys:
            groups.setdefault(self.shard_of(k.user_id), []).append(k)
        for shard_idx, sub_keys in groups.items():
            total += self.stores[shard_idx].delete(sub_keys).deleted
        return DeleteResult(deleted=total)

    def ttl(self, key: KVKey, ttl_seconds: int) -> None:
        self._store(key).ttl(key, ttl_seconds)

    def prefetch(self, keys: list[KVKey], *, dest: str = "hbm") -> list[Any]:
        groups: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(self.shard_of(k.user_id), []).append(i)
        out: list[Any] = [None] * len(keys)
        for shard_idx, idxs in groups.items():
            sub_keys = [keys[i] for i in idxs]
            for i, r in zip(idxs, self.stores[shard_idx].prefetch(sub_keys, dest=dest)):
                out[i] = r
        return out


def build_sharded_kv_store(
    configs: list[KVConfig],
    router: Router | None = None,
) -> ShardedKVStore:
    """按后端配置构建 N 个底层 store，再包成一致性哈希分片门面。"""
    stores = [build_kv_store(c) for c in configs]
    return ShardedKVStore(stores, router=router)