"""本地模拟 KV 后端（单卡/单机参照实现）。

作为 ``KVStore`` 协议的参考实现：内存 dict + 可选 mmap/文件持久化。
与 yuanrong adapter 共享同一套序列化与语义，保证「先单卡跑通，再切 datasystem」无缝。
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from onetrans.serving.kv_store import (
    AppendResult,
    DeleteResult,
    DeltaKV,
    KVKey,
    KVStore,
    PutResult,
    UserKVRecord,
)
from onetrans.serving.serialize import deserialize, serialize


def _cat_pair(
    a: tuple[Any, Any], b: tuple[Any, Any]
) -> tuple[Any, Any]:
    import torch

    return torch.cat([a[0], b[0]], dim=1), torch.cat([a[1], b[1]], dim=1)


class LocalKVStore(KVStore):
    """进程内 dict 后端，供数值/契约/时延基准。"""

    def __init__(self, dtype: str = "float16", **_: Any) -> None:
        self._dtype = dtype
        self._store: dict[str, UserKVRecord] = {}
        self._lock = threading.Lock()

    # -- 生命周期 ---------------------------------------------------------- #
    def connect(self, conf: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

    # -- 单对象读写 -------------------------------------------------------- #
    def put(self, rec: UserKVRecord) -> PutResult:
        with self._lock:
            self._store[str(rec.key)] = rec
        return PutResult(accepted=True, version=rec.key.model_version, checksum=rec.checksum)

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> Optional[UserKVRecord]:
        rec = self._store.get(str(key))
        if rec is None or layers is None:
            return rec
        # 只抽取指定层
        per_layer = deserialize(rec.payload)
        selected = [per_layer[l] for l in layers if l < len(per_layer)]
        if not selected:
            return None
        return _rebuild(rec, selected)

    def mget(self, keys: list[KVKey], *, layers: list[int] | None = None) -> list[Optional[UserKVRecord]]:
        return [self.get(k, layers=layers) for k in keys]

    # -- 增量 append ------------------------------------------------------- #
    def append(self, delta: DeltaKV) -> AppendResult:
        with self._lock:
            rec = self._store.get(str(delta.key))
            if rec is None:
                return AppendResult(False, 0, "", reason="missing")
            if delta.offset != rec.s_len:
                return AppendResult(False, rec.s_len, "", reason="offset_conflict")
            per_layer = deserialize(rec.payload)
            if len(per_layer) != len(delta.tensors):
                return AppendResult(False, rec.s_len, "", reason="layer_mismatch")
            merged = [_cat_pair(p, d) for p, d in zip(per_layer, delta.tensors)]
            new_rec = _rebuild(rec, merged, s_len=rec.s_len + delta.delta_len)
            self._store[str(delta.key)] = new_rec
            return AppendResult(True, new_rec.s_len, new_rec.checksum)

    # -- 生命周期/失效 ----------------------------------------------------- #
    def delete(self, keys: list[KVKey]) -> DeleteResult:
        n = 0
        with self._lock:
            for k in keys:
                if self._store.pop(str(k), None) is not None:
                    n += 1
        return DeleteResult(deleted=n)

    def ttl(self, key: KVKey, ttl_seconds: int) -> None:
        # 本地实现不落 TTL（仅记录语义），可扩展为定时清理
        return None

    def prefetch(self, keys: list[KVKey], *, dest: str = "hbm") -> list[Any]:
        # 本地后端无介质迁移语义，no-op 返回记录引用
        return [self._store.get(str(k)) for k in keys]


def _rebuild(
    rec: UserKVRecord,
    per_layer: list[tuple[Any, Any]],
    s_len: int | None = None,
) -> UserKVRecord:
    """用新的逐层张量重建记录（保持 key/版本/时间戳，重算长度与 checksum）。"""
    return UserKVRecord(
        key=rec.key,
        s_len=rec.s_len if s_len is None else s_len,
        per_layer_len=[k.shape[1] for k, _ in per_layer],
        dtype=rec.dtype,
        payload=serialize(per_layer),
        seq_ts_last=rec.seq_ts_last,
        created_at=rec.created_at,
    )