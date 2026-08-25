"""本地模拟 KV 后端（单卡/单机参照实现）。

作为 :class:`KVStore` 协议的参考实现：内存 dict + 可选 mmap 文件持久化。

- 默认（无 ``mmap_dir``）：payload 存 ``bytes``，读写都要经过一次字节缓冲。
- ``mmap_dir`` 指定后：payload 落盘并 ``mmap(ACCESS_READ`` 映射，record 暴露 ``memoryview``，
  上层经 ``serialize.deserialize`` 的 ``torch.frombuffer`` 直接视图共享内存，
  实现 **UserKV 读侧零拷贝**（消除 RAM 二次拷贝）。

与 yuanrong adapter 共享同一套序列化与语义，保证「先单卡跑通，再切 datasystem」无缝。
"""

from __future__ import annotations

import hashlib
import mmap
import os
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


def _cat_pair(a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
    import torch

    return torch.cat([a[0], b[0]], dim=1), torch.cat([a[1], b[1]], dim=1)


class LocalKVStore(KVStore):
    """进程内 dict 后端（可选 mmap 持久化 + 读侧零拷贝）。"""

    def __init__(self, dtype: str = "float16", mmap_dir: str | None = None, **_: Any) -> None:
        self._dtype = dtype
        self._store: dict[str, UserKVRecord] = {}
        self._lock = threading.Lock()
        self._mmap_dir = mmap_dir
        self._mmaps: dict[str, mmap.mmap] = {}
        if mmap_dir is not None:
            os.makedirs(mmap_dir, exist_ok=True)

    # -- 生命周期 ---------------------------------------------------------- #
    def connect(self, conf: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

    # -- 单对象读写 -------------------------------------------------------- #
    def put(self, rec: UserKVRecord) -> PutResult:
        with self._lock:
            self._persist(rec)
        return PutResult(accepted=True, version=rec.key.model_version, checksum=rec.checksum)

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> Optional[UserKVRecord]:
        rec = self._store.get(str(key))
        if rec is None or layers is None:
            return rec
        # 只抽取指定层（读取为 frombuffer 视图，无整对象反序列化拷贝）
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
            self._persist(new_rec)
            return AppendResult(True, new_rec.s_len, new_rec.checksum)

    # -- 生命周期/失效 ----------------------------------------------------- #
    def delete(self, keys: list[KVKey]) -> DeleteResult:
        n = 0
        with self._lock:
            for k in keys:
                key = str(k)
                if self._store.pop(key, None) is not None:
                    n += 1
                mm = self._mmaps.pop(key, None)
                if mm is not None:
                    mm.close()
        return DeleteResult(deleted=n)

    def ttl(self, key: KVKey, ttl_seconds: int) -> None:
        # 本地实现不落 TTL（仅记录语义），可扩展为定时清理
        return None

    def prefetch(self, keys: list[KVKey], *, dest: str = "hbm") -> list[Any]:
        # 本地后端无介质迁移语义，返回记录引用（mmap 模式下即共享内存视图）
        return [self._store.get(str(k)) for k in keys]

    # -- 内部：落盘 + mmap 视图 ------------------------------------------- #
    def _persist(self, rec: UserKVRecord) -> None:
        """写入存储；mmap 模式下把 payload 映射为共享内存视图（零拷贝读）。"""
        key = str(rec.key)
        if self._mmap_dir is None:
            self._store[key] = rec
            return
        # 关闭旧映射（若有），避免后续 truncate 触发 SIGBUS
        old = self._mmaps.pop(key, None)
        if old is not None:
            old.close()

        path = self._mmap_file(key, rec)
        with open(path, "wb") as f:
            f.write(rec.payload)
        with open(path, "r+b") as f:
            # ACCESS_WRITE：可写共享映射，frombuffer 视图可双向零拷贝（Linux 下关闭 fd 后映射仍有效）
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        self._mmaps[key] = mm
        # 暴露 memoryview：deserialize 的 frombuffer 直接视图 mmap，无拷贝
        rec.payload = memoryview(mm)
        self._store[key] = rec

    def _mmap_file(self, key: str, rec: UserKVRecord) -> str:
        # 文件名含 payload 校验和：同内容幂等复用，不同内容新文件（避免截断已映射文件）
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        csum = rec.checksum[:16]
        assert self._mmap_dir is not None
        return os.path.join(self._mmap_dir, f"{digest}-{csum}.bin")


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