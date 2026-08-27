"""Stage II 计算桥（PyTorch 算子下发，§7.4.5）。

由 C++ PythonComputeBridge 进程内嵌入调用（嵌入式解释器，非子进程）：

- ``init(weights_dir)``：解析 weights.bin（与 C++ ArtifactStore 同一格式）→ 构建
  OneTrans backbone + TwoStageRunner，权重常驻 GPU（无 CUDA 时 CPU）。
- ``score_batch(kv_blobs, row_map, ns_blob, n_rows, ns, d)``：批量交叉注意力前向
  （C++ 已完成 lookup/encode_ns/KV mget/攒批，本函数只做最终计算下发）。

数值契约：与 C++ ``TwoStageRunner::score_ns_batch`` 逐位对拍（e2e_test 守护，
golden 阈值 1e-5）。UserKV payload 为 C++/Python 二进制兼容格式（serialize.py）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from onetrans.models.one_trans import OneTrans
from onetrans.serving.serialize import deserialize_with_meta
from onetrans.serving.two_stage import TwoStageRunner, UserKV

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_runner: TwoStageRunner | None = None
_META: dict = {}


def _load_state_dict(weights_dir: Path) -> dict[str, torch.Tensor]:
    manifest = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))
    blob = np.fromfile(weights_dir / "weights.bin", dtype="<f4")
    sd: dict[str, torch.Tensor] = {}
    for name, meta in manifest["tensors"].items():
        n = 1
        for s in meta["shape"]:
            n *= s
        flat = blob[meta["offset"] : meta["offset"] + n].copy()
        sd[name] = torch.from_numpy(flat).view(*meta["shape"])
    return sd


def init(weights_dir: str) -> dict:
    """加载权重并构建 runner（幂等；返回运行元数据供 C++ 侧校验）。"""
    global _runner, _META

    wdir = Path(weights_dir)
    cfg = json.loads((wdir / "manifest.json").read_text(encoding="utf-8"))["config"]
    sd = _load_state_dict(wdir)

    backbone = OneTrans(
        d_model=cfg["d_model"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
        max_seq_len=cfg["max_seq_len"],
        min_seq_len=cfg["min_seq_len"],
        ns_tokens_num=cfg["ns_tokens_num"],
        dropout=0.0,
        use_cls_token=cfg["use_cls_token"],
    )
    # manifest 中 backbone.* 前缀即本模块 state_dict 键（dims 为普通属性不进 state_dict）
    backbone_sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"bridge: 权重不完整 missing={missing} unexpected={unexpected}")
    backbone = backbone.to(_DEVICE).eval()

    _runner = TwoStageRunner(backbone)
    _META = {
        "device": _DEVICE,
        "d_model": cfg["d_model"],
        "num_blocks": cfg["num_blocks"],
        "ns_tokens_num": cfg["ns_tokens_num"],
        "dims": cfg["dims"],
    }
    return dict(_META)


def meta() -> dict:
    return dict(_META)


def _kv_from_payload(payload: bytes) -> UserKV:
    per_layer, s_len, per_layer_len = deserialize_with_meta(payload)
    return UserKV(
        per_layer=[(k.to(_DEVICE), v.to(_DEVICE)) for k, v in per_layer],
        per_layer_len=per_layer_len,
        s_len=s_len,
    )


def score_batch(
    kv_blobs: list[bytes],
    row_map: list[int],
    ns_blob: bytes,
    n_rows: int,
    ns: int,
    d: int,
) -> bytes:
    """批量 Stage II 前向。

    :param kv_blobs: 去重后的 UserKV payload 列表（C++ kv_serialize 格式）
    :param row_map: 每候选行 → kv_blobs 下标（长度 == n_rows；同用户多候选共享 KV）
    :param ns_blob: fp32 小端连续字节，[n_rows, ns, d]（C++ Frontend Encode Pool 产出）
    :return: logits fp32 小端字节 [n_rows, T]
    """
    if _runner is None:
        raise RuntimeError("bridge: 未初始化（先调 init）")

    ns_emb = (
        torch.frombuffer(bytearray(ns_blob), dtype=torch.float32)
        .reshape(n_rows, ns, d)
        .to(_DEVICE)
    )

    cache: dict[int, UserKV] = {}
    kvs: list[UserKV] = []
    for idx in row_map:
        kv = cache.get(idx)
        if kv is None:
            kv = _kv_from_payload(kv_blobs[idx])
            cache[idx] = kv
        kvs.append(kv)

    with torch.no_grad():
        logits = _runner.score_ns_batch(kvs, ns_emb)
    return logits.detach().to("cpu").contiguous().numpy().tobytes()
