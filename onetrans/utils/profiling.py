import time
from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(modules: Dict[str, nn.Module]) -> Dict[str, int]:
    """Per-module and total parameter counts (N in scaling-law notation)."""
    metrics: Dict[str, int] = {}
    total = 0
    trainable = 0
    for name, module in modules.items():
        n = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        metrics[f"params/{name}"] = n
        total += n
        trainable += n_trainable
    metrics["params/total"] = total
    metrics["params/trainable"] = trainable
    return metrics


def measure_forward_flops(forward_fn: Callable[[], object]) -> int:
    """Total FLOPs of a single forward pass, measured at the aten-op level."""
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        forward_fn()
    return counter.get_total_flops()


def measure_train_step_flops(step_fn: Callable[[], object]) -> int:
    """Total FLOPs of one forward + backward pass (true per-step training FLOPs)."""
    counter = FlopCounterMode(display=False)
    with counter:
        step_fn()
    return counter.get_total_flops()


@torch.no_grad()
def measure_inference(
    forward_fn: Callable[[], object],
    device: torch.device,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> Dict[str, float]:
    """Median per-batch inference latency (ms) and peak memory (MB)."""
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(n_warmup):
        forward_fn()
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_iters):
        start = time.perf_counter()
        forward_fn()
        if is_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    median_ms = timings[len(timings) // 2]
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda else 0.0
    return {
        "inference/latency_ms": median_ms,
        "inference/peak_memory_mb": peak_mem_mb,
    }


def profile_model(
    modules: Dict[str, nn.Module],
    forward_fn: Callable[[], object],
    train_step_fn: Callable[[], object],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """One-off profile: params, FLOPs (fwd + train step), inference latency/memory."""
    was_training = {name: m.training for name, m in modules.items()}
    for m in modules.values():
        m.eval()

    metrics: Dict[str, float] = {}
    metrics.update(count_parameters(modules))

    fwd_flops = measure_forward_flops(forward_fn)
    metrics["flops/forward_per_batch"] = fwd_flops
    metrics["flops/forward_per_sample"] = fwd_flops / max(batch_size, 1)

    train_flops = measure_train_step_flops(train_step_fn)
    metrics["flops/train_step_per_batch"] = train_flops
    metrics["flops/train_step_per_sample"] = train_flops / max(batch_size, 1)

    metrics.update(measure_inference(forward_fn, device))

    for name, m in modules.items():
        m.train(was_training[name])
    return metrics
