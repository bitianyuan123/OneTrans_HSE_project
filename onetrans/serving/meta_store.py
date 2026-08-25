"""元数据/版本面（fabric ③）：KVPointer + TTL/LRU 失效。

存储「一用户、一模型版本」的轻量指针（版本/长度/时间戳/校验和/存储地址），供：

- Nearline 写侧记录 append 后的新指针；
- Online 读侧校验 ``record.checksum == pointer.checksum``（不一致触发兜底/降级）；
- 路由/失效面做 TTL 淘汰与 LRU 本地副本管理。

与 payload 分离（见设计文档 §1.1 原则 3、§3.4 元数据面），元数据只存结构化小对象。
"""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from onetrans.serving.kv_store import KVKey, UserKVRecord


def pointer_key(model_version: str, user_id: str) -> str:
    """元数据主键（与 §1.6 的 ``meta:{mv}:{uid}`` 对齐）。"""
    return f"meta:{KVKey(model_version, user_id)}"  # 复用 KVKey 的字符集规范化


@dataclass
class KVPointer:
    """User KV 的元数据指针（结构化小对象，不存储 payload）。"""

    model_version: str
    user_id: str
    checksum: str  # payload sha256（内容指纹）
    s_len: int
    per_layer_len: list[int]
    seq_ts_last: int = 0
    obj_key: str = ""  # 存储层键/地址（datasystem/本地文件/S3 key）
    created_at: int = 0

    @property
    def key(self) -> str:
        return pointer_key(self.model_version, self.user_id)


def validate_pointer(rec: UserKVRecord, ptr: KVPointer | None) -> bool:
    """校验「读到的 record」与「元数据 pointer」一致（设计文档 §5 的读侧一致性）。"""
    if ptr is None:
        return False
    return (
        rec.checksum == ptr.checksum
        and rec.s_len == ptr.s_len
        and rec.per_layer_len == ptr.per_layer_len
    )


class MetaStore(Protocol):
    """元数据面统一接口（存储无关）。"""

    def get(self, model_version: str, user_id: str) -> Optional[KVPointer]: ...

    def get_multi(self, keys: list[KVKey]) -> list[Optional[KVPointer]]: ...

    def set(self, ptr: KVPointer) -> None: ...

    def delete(self, keys: list[KVKey]) -> int: ...

    def ttl(self, model_version: str, user_id: str, ttl_seconds: int) -> None: ...


class LocalMetaStore:
    """进程内元数据后端：dict + 惰性 TTL 过期 + 线程安全。"""

    def __init__(self) -> None:
        self._ptrs: dict[str, KVPointer] = {}
        self._expire: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, model_version: str, user_id: str) -> Optional[KVPointer]:
        key = pointer_key(model_version, user_id)
        with self._lock:
            self._lazy_expire(key)
            return self._ptrs.get(key)

    def get_multi(self, keys: list[KVKey]) -> list[Optional[KVPointer]]:
        return [self.get(k.model_version, k.user_id) for k in keys]

    def set(self, ptr: KVPointer) -> None:
        with self._lock:
            self._ptrs[ptr.key] = ptr
            self._expire.pop(ptr.key, None)

    def delete(self, keys: list[KVKey]) -> int:
        n = 0
        with self._lock:
            for k in keys:
                key = pointer_key(k.model_version, k.user_id)
                if self._ptrs.pop(key, None) is not None:
                    n += 1
                self._expire.pop(key, None)
        return n

    def ttl(self, model_version: str, user_id: str, ttl_seconds: int) -> None:
        key = pointer_key(model_version, user_id)
        with self._lock:
            if key in self._ptrs:
                self._expire[key] = time.monotonic() + ttl_seconds

    def _lazy_expire(self, key: str) -> None:
        exp = self._expire.get(key)
        if exp is not None and time.monotonic() >= exp:
            self._ptrs.pop(key, None)
            self._expire.pop(key, None)

    def size(self) -> int:
        with self._lock:
            return len(self._ptrs)


@dataclass
class LocalMetaConfig:
    """本地元数据后端配置占位（可扩展容量/LRU）。"""

    max_entries: int = 1_000_000
    default_ttl: int = 0  # 0 = 不过期


def build_meta_store(conf: Any = None) -> MetaStore:
    """元数据面工厂（当前仅本地实现，生产可切 Redis Cluster）。"""
    return LocalMetaStore()