"""Nearline/Online 编排（两阶段服务 worker）。

- :class:`NearlineWorker`：Stage I，编码用户历史并写入 User KV（prefill）。
- :class:`OnlineWorker`：Stage II，读 User KV 并对候选交叉打分（decode）。
- :class:`BatchScheduler`：FIFO 攒批调度器，超时/满批触发批量打分（吞吐优先）。

worker 只依赖 :class:`TwoStageRunner`、:class:`KVStore` 与 :class:`ServingMetrics`，
与底层存储后端解耦（见 ``kv_store.py`` 的存储无关契约）。

增量 append 优化（复用旧 K/V 仅算新增位置）为后续里程碑；当前 nearline 采用
「全量 prefill + put」的正确基线（对应设计文档 §3.1 的幂等重建路径）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
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
            payload = serialize(kv.per_layer, s_len=kv.s_len, per_layer_len=kv.per_layer_len)
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
        self._num_tasks = self.runner.linear.out_features

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
            # KV miss（冷启动/过期/搬迁）：返回全零 legal logits，不抛异常（见 gap_analysis G8）
            self.metrics.count("kv.miss", 1.0)
            return torch.zeros(ns_emb.shape[0], self._num_tasks, device=ns_emb.device)
        self.metrics.count("kv.hit", 1.0)

        with self.metrics.timing("online.encode_stage2"):
            kv = decode_record(rec, ns_emb.device)
            logits = self.runner.score_ns(kv, ns_emb)
        self.metrics.count("online.qps", 1.0)
        self.metrics.count("online.candidate_throughput", float(ns_emb.shape[0]))
        return logits

    def score_batch(self, batch: list[ScoreRequest]) -> Tensor:
        """批量打分：一次 ``mget`` + 一次 ``score_ns_batch`` 打包多个请求。

        :param batch: 若干（user, 候选）打分请求（:class:`ScoreRequest`）
        :return: logits ``[B, T]``（B 为所有候选总数，顺序与 batch 展平一致）
        """
        keys = [r.key for r in batch]
        with self.metrics.timing("online.kv_mget"):
            recs = self.store.mget(keys)
        kvs: list[Any] = []
        embs: list[Tensor] = []
        miss_positions: list[int] = []
        flat = 0
        for rec, req in zip(recs, batch):
            m = req.ns_emb.shape[0]
            if rec is None:
                self.metrics.count("kv.miss", 1.0)
                miss_positions.extend(range(flat, flat + m))  # 记录 miss 候选的展平位置
            else:
                self.metrics.count("kv.hit", 1.0)
                kv = decode_record(rec, req.ns_emb.device)
                # 同用户多候选：kv 重复、候选展平
                for _ in range(m):
                    kvs.append(kv)
                embs.append(req.ns_emb)
            flat += m
        total = flat

        # miss 降级：命中候选正常打分，miss 候选填全零，保持 [B, T] 与展平顺序（见 gap_analysis G8）
        out = torch.zeros(total, self._num_tasks, device=batch[0].ns_emb.device)
        if kvs:
            ns_emb = torch.cat(embs, dim=0)  # [num_hits, Ns, D]
            with self.metrics.timing("online.encode_stage2"):
                logits = self.runner.score_ns_batch(kvs, ns_emb)  # [num_hits, T]
            miss_set = set(miss_positions)
            hit_ptr = 0
            for i in range(total):
                if i not in miss_set:
                    out[i] = logits[hit_ptr]
                    hit_ptr += 1
        self.metrics.count("online.qps", float(len(batch)))
        self.metrics.count("online.candidate_throughput", float(total))
        self.metrics.gauge("online.batch_size", float(total))
        return out


@dataclass
class ScoreRequest:
    """一次打分请求（一个 user 的 M 个候选）。"""

    key: KVKey
    ns_emb: Tensor  # [M, Ns, D]


class BatchScheduler:
    """FIFO 攒批调度器：满批或超时即吐出一个批次（动态 batching / 吞吐优先）。

    线程安全：生产者 :meth:`submit` 入队，消费者 :meth:`next_batch` 阻塞攒批。
    攒批窗口 ``max_wait_seconds`` 内若未满批，也按「已攒到的 ≥1 条」返回，
    保证时延有界。
    """

    def __init__(self, max_batch_size: int = 64, max_wait_seconds: float = 0.005) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须 ≥ 1")
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._queue: deque[ScoreRequest] = deque()
        self._cond = threading.Condition()

    def submit(self, req: ScoreRequest) -> None:
        with self._cond:
            self._queue.append(req)
            self._cond.notify()

    def next_batch(self) -> list[ScoreRequest]:
        """阻塞取出一个批次（≥1 条，满批或超时返回）。"""
        with self._cond:
            while not self._queue:
                self._cond.wait()
            deadline = time.monotonic() + self.max_wait_seconds
            while len(self._queue) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            batch: list[ScoreRequest] = []
            while self._queue and len(batch) < self.max_batch_size:
                batch.append(self._queue.popleft())
            return batch

    def __len__(self) -> int:
        with self._cond:
            return len(self._queue)


def decode_record(rec: UserKVRecord, device: torch.device) -> Any:
    """把 UserKVRecord 反序列化为 :class:`UserKV`（供二阶段引擎直接消费）。"""
    from onetrans.serving.two_stage import UserKV

    per_layer = [(k.to(device), v.to(device)) for k, v in deserialize(rec.payload)]
    return UserKV(per_layer=per_layer, per_layer_len=rec.per_layer_len, s_len=rec.s_len)