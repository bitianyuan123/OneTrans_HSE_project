"""计算面线程模型（P1）：Dispatcher + WorkerPool + req_seq 异步匹配 + 背压。

这是 C++ 侧基于 brpc + bthread 的 M:N 线程模型的**单机 Python 参照实现**，刻画同一套
并发原语，用于「先在单机跑通并发语义与数值一致，再映射 brpc」：

- :class:`Dispatcher`：分配单调递增 ``req_seq``，按 ``user_id`` 稳定哈希（数据本地化）
  或轮询把请求派发到 worker；
- :class:`WorkerPool`：N 个 worker 线程，每个持有**独立有界队列**（消除全局锁，
  避免单锁串行化 batch 提交，QPS 才能上去）；
- req_seq **异步匹配**：响应携带 ``req_seq``，经 :class:`concurrent.futures.Future`
  匹配回调用方，支持乱序完成；
- **背压**：worker 队列满时拒绝（:class:`OverloadRejected`，可重试/降级）或限时阻塞，
  避免无界排队拖垮 p99 时延。

对应设计文档 ``docs/detailed_design.md`` §7.4 的「线程与并发模型」；生产 C++ 对应物为
``brpc::Server + bthread + ExecutionQueue/ThreadPool``。
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

from onetrans.serving.router import Router

_STOP = object()


@dataclass
class Request:
    """一次计算请求（携带由 Dispatcher 分配的 req_seq）。"""

    user_id: str
    payload: Any
    req_seq: int = -1


@dataclass
class Response:
    """异步响应，req_seq 用于匹配回调用方（乱序完成）。"""

    req_seq: int
    user_id: str
    result: Any
    worker_id: int = -1


class OverloadRejected(Exception):
    """背压信号：worker 队列满，请求被拒绝（调用方可重试 / 降级 / 打点）。"""

    def __init__(self, req_seq: int) -> None:
        super().__init__(f"overload rejected req_seq={req_seq}")
        self.req_seq = req_seq


class WorkerPool:
    """N 个 worker 线程 + 各自独立有界队列（无全局锁的并发执行面）。"""

    def __init__(
        self,
        num_workers: int,
        queue_capacity: int,
        handler: Callable[[Request], Any],
        router: Optional[Router] = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers 必须 ≥ 1")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity 必须 ≥ 1")
        self.num_workers = num_workers
        # 统一复用 Router（jump 哈希）：worker 分派与 KV 分片对同一 user 落同一桶，
        # 保证数据本地性（见 gap_analysis G3）；默认 num_shards == num_workers。
        self.router = router or Router(num_shards=num_workers)
        self._queues: list[queue.Queue] = [
            queue.Queue(maxsize=queue_capacity) for _ in range(num_workers)
        ]
        self._workers: list[threading.Thread] = []
        self._on_done: Optional[Callable[[Request, Any, int], None]] = None
        self._shutdown = False
        self._started = False
        for i in range(num_workers):
            t = threading.Thread(
                target=self._run, args=(i, self._queues[i], handler), daemon=True,
            )
            self._workers.append(t)

    def set_on_done(self, cb: Callable[[Request, Any, int], None]) -> None:
        self._on_done = cb

    def start(self) -> None:
        self._started = True
        for t in self._workers:
            t.start()

    def stop(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if not self._started:
            return  # worker 线程从未启动，无消费方、无需哨兵/join
        for q in self._queues:
            q.put(_STOP)
        for t in self._workers:
            t.join()

    # -- 路由：复用 Router（jump 哈希），与 KV 分片 locality 对齐 ---------- #
    def worker_for(self, key: str) -> int:
        return self.router.route(key)

    # -- 入队：满队 → 背压（尝试或限时阻塞） -------------------------------- #
    def try_enqueue(self, worker_id: int, req: Request) -> bool:
        try:
            self._queues[worker_id].put_nowait(req)
            return True
        except queue.Full:
            return False

    def enqueue(self, worker_id: int, req: Request, timeout: float) -> bool:
        try:
            self._queues[worker_id].put(req, timeout=timeout)
            return True
        except queue.Full:
            return False

    def pending(self) -> int:
        return sum(q.qsize() for q in self._queues)

    # -- worker 主循环 ---------------------------------------------------- #
    def _run(self, worker_id: int, q: queue.Queue, handler: Callable[[Request], Any]) -> None:
        while True:
            req = q.get()
            if req is _STOP:
                return
            try:
                result = handler(req)
            except Exception as exc:  # noqa: BLE001 计算异常原样回传，由 Future 承载
                result = exc
            if self._on_done is not None:
                self._on_done(req, result, worker_id)


class Dispatcher:
    """分配 req_seq、派发 worker、经 Future 异步匹配响应（支持乱序完成）。"""

    def __init__(
        self,
        pool: WorkerPool,
        mode: str = "hash",  # hash（数据本地）| round_robin
        backpressure_timeout: Optional[float] = None,
    ) -> None:
        if mode not in ("hash", "round_robin"):
            raise ValueError("mode 须为 hash 或 round_robin")
        self.pool = pool
        self.mode = mode
        self.backpressure_timeout = backpressure_timeout

        self._seq = 0
        self._seq_lock = threading.Lock()
        self._inflight: dict[int, Future] = {}
        self._inflight_lock = threading.Lock()
        self._rr = 0
        self._rr_lock = threading.Lock()

        pool.set_on_done(self._complete)

    # -- 派发 ------------------------------------------------------------ #
    def submit(self, user_id: str, payload: Any, timeout: Optional[float] = None) -> Future:
        """提交一个请求并返回 :class:`Future`（result 为 :class:`Response`）。

        队列满时视 ``timeout``/``backpressure_timeout`` 决定「限时阻塞」或「直接拒绝」，
        拒绝即在 Future 上 set :class:`OverloadRejected`（背压，不抛同步异常）。
        """
        seq = self._alloc_seq()
        fut: Future = Future()
        with self._inflight_lock:
            self._inflight[seq] = fut

        req = Request(user_id=user_id, payload=payload, req_seq=seq)
        worker_id = self._choose_worker(user_id)
        t = timeout if timeout is not None else self.backpressure_timeout
        ok = (
            self.pool.enqueue(worker_id, req, t)
            if t is not None and t > 0
            else self.pool.try_enqueue(worker_id, req)
        )
        if not ok:
            with self._inflight_lock:
                self._inflight.pop(seq, None)
            if not fut.done():
                fut.set_exception(OverloadRejected(seq))
        return fut

    def inflight(self) -> int:
        with self._inflight_lock:
            return len(self._inflight)

    def close(self) -> None:
        self.pool.stop()

    # -- 内部 ------------------------------------------------------------ #
    def _alloc_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _choose_worker(self, user_id: str) -> int:
        if self.mode == "hash":
            return self.pool.worker_for(user_id)
        with self._rr_lock:
            w = self._rr
            self._rr = (self._rr + 1) % self.pool.num_workers
            return w

    def _complete(self, req: Request, result: Any, worker_id: int) -> None:
        fut: Optional[Future] = None
        with self._inflight_lock:
            fut = self._inflight.pop(req.req_seq, None)
        if fut is None or fut.done():
            return
        if isinstance(result, Exception):
            fut.set_exception(result)
        else:
            fut.set_result(
                Response(req_seq=req.req_seq, user_id=req.user_id, result=result, worker_id=worker_id)
            )