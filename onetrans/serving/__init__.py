"""序列 Transformer 精排（OneTrans 类）推理服务层。

对应设计文档 ``docs/detailed_design.md``：
- ``kv_store``         KV 存储接口契约（存储无关）
- ``serialize``        UserKV payload 序列化
- ``local_adapter``    本地模拟后端
- ``datasystem_adapter`` yuanrong datasystem 后端
- ``two_stage``        两阶段推理引擎（encode_s / score_ns）
- ``metrics``          指标采集
- ``pipeline``         Nearline/Online 编排
- ``router``           一致性哈希路由（user → owner shard）
- ``meta_store``       元数据/版本面（KVPointer + TTL/LRU 失效）
- ``sharded``          一致性哈希分片 KVStore（数据本地化）
"""

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
from onetrans.serving.meta_store import KVPointer, MetaStore, build_meta_store
from onetrans.serving.metrics import ServingMetrics
from onetrans.serving.pipeline import NearlineWorker, OnlineWorker
from onetrans.serving.router import JumpConsistentHash, RingHash, Router, hash64
from onetrans.serving.sharded import ShardedKVStore, build_sharded_kv_store
from onetrans.serving.two_stage import TwoStageRunner

__all__ = [
    "TwoStageRunner",
    "NearlineWorker",
    "OnlineWorker",
    "ServingMetrics",
    "KVKey",
    "KVConfig",
    "KVStore",
    "UserKVRecord",
    "DeltaKV",
    "PutResult",
    "AppendResult",
    "DeleteResult",
    "build_kv_store",
    "KVPointer",
    "MetaStore",
    "build_meta_store",
    "JumpConsistentHash",
    "RingHash",
    "Router",
    "hash64",
    "ShardedKVStore",
    "build_sharded_kv_store",
]