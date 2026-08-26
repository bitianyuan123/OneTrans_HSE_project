"""KV 存储接口契约（存储无关）。

本模块只定义**逻辑契约**，不依赖任何具体后端。上层（Nearline/Online Worker）
只 import 本模块的抽象类型；具体后端通过实现 :class:`KVStore` 协议接入
（本地模拟 / yuanrong datasystem / Redis+S3 等）。

与设计文档的对应关系见 ``docs/detailed_design.md`` 第 1 节。
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

import torch


def _urlsafe_b64(s: str) -> str:
    """url-safe base64（去 ``=`` 填充），用于 datasystem key 字符集受限场景（§1.6）。"""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# 键与结果类型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KVKey:
    """User KV 的逻辑主键。"""

    model_version: str
    user_id: str

    def __str__(self) -> str:  # datasystem KV key 规范（字符集受限，见设计文档 §1.6）
        # user_id 做 url-safe base64，避免越界字符
        import base64

        uid = (
            base64.urlsafe_b64encode(str(self.user_id).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        mv = base64.urlsafe_b64encode(self.model_version.encode("utf-8")).decode("ascii").rstrip("=")
        return f"kv:{mv}:{uid}"


@dataclass
class PutResult:
    accepted: bool
    version: str
    checksum: str
    reason: str = ""


@dataclass
class AppendResult:
    accepted: bool
    new_s_len: int
    checksum: str
    reason: str = ""


@dataclass
class DeleteResult:
    deleted: int


@dataclass
class DeltaKV:
    """增量 append 的载荷：在 offset 处追加 ``delta_len`` 个 S token 的逐层 K/V。"""

    key: KVKey
    base_version: str
    offset: int  # 期望等于当前对象 s_len（乐观并发校验用）
    delta_len: int
    tensors: list[tuple[torch.Tensor, torch.Tensor]]  # per-layer (ΔK_s^l, ΔV_s^l)
    expect_checksum: str = ""  # fencing token：非空时要求当前 payload checksum 匹配（见 gap_analysis G2）


# --------------------------------------------------------------------------- #
# 逻辑对象模型
# --------------------------------------------------------------------------- #
@dataclass
class UserKVRecord:
    """一用户、一模型版本的整套逐层 K/V（序列化形态，跨存储边界）。"""

    key: KVKey
    s_len: int
    per_layer_len: list[int]
    dtype: str
    payload: bytes
    seq_ts_last: int = 0
    created_at: int = 0

    @property
    def checksum(self) -> str:
        return compute_checksum(self.payload)


def compute_checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# 统一接口契约
# --------------------------------------------------------------------------- #
@runtime_checkable
class KVStore(Protocol):
    """User KV 的统一逻辑接口（存储无关）。见设计文档 §1.3。"""

    def connect(self, conf: dict[str, Any] | None = None) -> None: ...

    def close(self) -> None: ...

    def put(self, rec: UserKVRecord) -> PutResult: ...

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> Optional[UserKVRecord]: ...

    def mget(self, keys: list[KVKey], *, layers: list[int] | None = None) -> list[Optional[UserKVRecord]]: ...

    def append(self, delta: DeltaKV) -> AppendResult: ...

    def delete(self, keys: list[KVKey]) -> DeleteResult: ...

    def ttl(self, key: KVKey, ttl_seconds: int) -> None: ...

    def prefetch(self, keys: list[KVKey], *, dest: str = "hbm") -> list[Any]:
        """可选：显式预取/迁移到目标介质（HBM/DRAM），返回零拷贝引用。"""


@dataclass
class KVConfig:
    """KVStore 的通用配置（后端无关的公共字段）。"""

    backend: str = "local"  # local | datasystem | redis
    dtype: str = "float16"  # 序列化默认 dtype
    local: dict[str, Any] = field(default_factory=dict)
    datasystem: dict[str, Any] = field(default_factory=dict)


def build_kv_store(conf: KVConfig) -> "KVStore":
    """按 ``backend`` 选择后端并连接（存储无关的工厂）。"""
    if conf.backend == "local":
        from onetrans.serving.local_adapter import LocalKVStore

        store: KVStore = LocalKVStore(dtype=conf.dtype, **conf.local)
    elif conf.backend == "datasystem":
        from onetrans.serving.datasystem_adapter import YuanrongKVStore

        store = YuanrongKVStore(dtype=conf.dtype, **conf.datasystem)
    elif conf.backend == "redis":
        raise NotImplementedError("redis 后端为等价 adapter，尚未实现")
    else:
        raise ValueError(f"unknown backend: {conf.backend}")
    store.connect(conf.datasystem if conf.backend == "datasystem" else conf.local)
    return store