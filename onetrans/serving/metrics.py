"""指标采集（轻量，无外部依赖）。

对应用设计文档 §6 的埋点命名约定 ``组件.阶段.子操作``。
提供三类原语：计时（histogram）、计数（counter）、仪表（gauge），
并提供 p50/p99 分位数快照，便于本地基准与后续接入 Prometheus/OTel。
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol


@dataclass
class _Histogram:
    samples: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]


class ServingMetrics:
    """收集时延直方图、计数器与仪表快照。"""

    def __init__(self) -> None:
        self._hist: defaultdict[str, _Histogram] = defaultdict(_Histogram)
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}

    # -- 原语 ------------------------------------------------------------ #
    @contextmanager
    def timing(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._hist[name].record((time.perf_counter() - start) * 1000.0)

    def count(self, name: str, n: float = 1.0) -> None:
        self._counters[name] += n

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    # -- 快照 ------------------------------------------------------------ #
    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, h in self._hist.items():
            out[f"{name}_p50"] = h.percentile(50)
            out[f"{name}_p99"] = h.percentile(99)
            out[f"{name}_count"] = float(len(h.samples))
        for name, v in self._counters.items():
            out[name] = v
        out.update(self._gauges)
        return out

    def clear(self) -> None:
        self._hist.clear()
        self._counters.clear()
        self._gauges.clear()


def report_table(snapshot: dict[str, float], title: str = "metrics") -> str:
    """把快照渲染成对齐的可读文本（用于 demo 输出）。"""
    lines = [f"[{title}]"]
    width = max((len(k) for k in snapshot), default=0) + 2
    for k in sorted(snapshot):
        lines.append(f"  {k:<{width}} {snapshot[k]:.4f}")
    return "\n".join(lines)


class MetricsSink(Protocol):
    """可选的指标出口协议（对接 Prometheus/OTel 时实现）。

    默认 :class:`ServingMetrics` 即满足该协议（本地内存收集）。
    """

    def timing(self, name: str) -> Callable[[], None]: ...
    def count(self, name: str, n: float = 1.0) -> None: ...
    def gauge(self, name: str, value: float) -> None: ...