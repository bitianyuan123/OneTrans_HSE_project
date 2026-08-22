"""Nearline/Online 编排（两阶段服务 worker）。

- :class:`NearlineWorker`：Stage I，编码用户历史并写入 User KV（prefill）。
- :class:`OnlineWorker`：Stage II，读 User KV 并对候选交叉打分（decode）。

worker 只依赖 :class:`TwoStageRunner`、:class:`KVStore` 与 :class:`ServingMetrics`，
与底层存储后端解耦（见 ``kv_store.py`` 的存储无关契约）。

增量 append 优化（复用旧 K/V 仅算新增位置）为后续里程碑；当前 nearline 采用
「全量 prefill + put」的正确基线（对应设计文档 §3.1 的幂等重建路径）。
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from onetrans.serving.kv_store import KVKey, KVStore, PutResult, UserKVRecord
from onetrans.serving.metrics import ServingMetrics
from onetrans.serving.serialize import deserialize, serialize
from onetrans.serving.two_stage import TwoStageRunner


class NearlineWorker:
    def __init__(
        self,
        runner: TwoStageRunner,
        store: KVStore,
        metrics: ServingMetrics | None = None,
        dtype: str = "float16",
    ) -> None:
        self.runner = runner
        self.store = store
        self.metrics = metrics or ServingMetrics()
        self.dtype = dtype

    def ingest(
        self,
        s_emb: Tensor,
        s_mask: Tensor,
        user_id: str,
        model_version: str,
        seq_ts_last: int = 0,
    ) -> PutResult:
        """编码用户历史并写入 User KV。

        :param s_emb: tokenize+pos+RMSNorm 后的 S 序列 [1, S0, D]
        :param s_mask: [1, S0] bool 有效掩码
        """
        with self.metrics.timing("nearline.encode_stage1"):
            kv = self.runner.encode_s(s_emb, s_mask)
        self.metrics.count("nearline.events_ingested", 1.0)

        with self.metrics.timing("nearline.append_kv"):
            payload = serialize(kv.per_layer)
            rec = UserKVRecord(
                key=KVKey(model_version=model_version, user_id=user_id),
                s_len=kv.s_len,
                per_layer_len=kv.per_layer_len,
                dtype=self.dtype,
                payload=payload,
                seq_ts_last=seq_ts_last,
                created_at=int(time.time()),
            )
            res = self.store.put(rec)
        if hasattr(self.store, "size"):
            self.metrics.gauge("kv.obj_cnt", float(self.store.size()))
        return res


class OnlineWorker:
    def __init__(
        self,
        runner: TwoStageRunner,
        store: KVStore,
        metrics: ServingMetrics | None = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.metrics = metrics or ServingMetrics()

    def score(
        self,
        user_id: str,
        model_version: str,
        ns_emb: Tensor,
    ) -> Tensor:
        """读 User KV 并对候选 NS token 打分。

        :param ns_emb: tokenize+RMSNorm 后的 NS 序列 [B=M, Ns, D]
        :return: logits [M, T]
        """
        key = KVKey(model_version=model_version, user_id=user_id)
        with self.metrics.timing("online.kv_get"):
            rec = self.store.get(key)
        if rec is None:
            self.metrics.count("kv.miss", 1.0)
            raise KeyError(f"user KV missing: {key}")
        self.metrics.count("kv.hit", 1.0)

        with self.metrics.timing("online.encode_stage2"):
            kv = decode_record(rec, ns_emb.device)
            logits = self.runner.score_ns(kv, ns_emb)
        self.metrics.count("online.qps", 1.0)
        self.metrics.count("online.candidate_throughput", float(ns_emb.shape[0]))
        return logits


def decode_record(rec: UserKVRecord, device: torch.device) -> Any:
    """把 UserKVRecord 反序列化为 :class:`UserKV`（供二阶段引擎直接消费）。"""
    from onetrans.serving.two_stage import UserKV

    per_layer = [(k.to(device), v.to(device)) for k, v in deserialize(rec.payload)]
    return UserKV(per_layer=per_layer, per_layer_len=rec.per_layer_len, s_len=rec.s_len)