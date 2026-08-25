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
- ``dispatcher``       计算面线程模型（Dispatcher + WorkerPool + 背压）
- ``embedding_ps_client`` 独立稀疏参数服务器（PS）数据面客户端
"""

from onetrans.serving.dispatcher import Dispatcher, OverloadRejected, Request, Response, WorkerPool
from onetrans.serving.embedding_ps_client import (
    EmbeddingPSClient,
    LocalEmbeddingPS,
    ShardedEmbeddingTable,
)
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
from onetrans.serving.pipeline import BatchScheduler, NearlineWorker, OnlineWorker, ScoreRequest
from onetrans.serving.router import JumpConsistentHash, RingHash, Router, hash64
from onetrans.serving.sharded import ShardedKVStore, build_sharded_kv_store
from onetrans.serving.two_stage import TwoStageRunner

__all__ = [
    "TwoStageRunner",
    "NearlineWorker",
    "OnlineWorker",
    "BatchScheduler",
    "ScoreRequest",
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
    "Dispatcher",
    "WorkerPool",
    "Request",
    "Response",
    "OverloadRejected",
    "EmbeddingPSClient",
    "LocalEmbeddingPS",
    "ShardedEmbeddingTable",
]