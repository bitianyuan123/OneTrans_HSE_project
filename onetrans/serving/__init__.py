"""序列 Transformer 精排（OneTrans 类）推理服务层。

对应设计文档 ``docs/detailed_design.md``：
- ``kv_store``         KV 存储接口契约（存储无关）
- ``serialize``        UserKV payload 序列化
- ``local_adapter``    本地模拟后端
- ``datasystem_adapter`` yuanrong datasystem 后端
- ``two_stage``        两阶段推理引擎（encode_s / score_ns）
- ``metrics``          指标采集
- ``pipeline``         Nearline/Online 编排
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
from onetrans.serving.metrics import ServingMetrics
from onetrans.serving.pipeline import NearlineWorker, OnlineWorker
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
]