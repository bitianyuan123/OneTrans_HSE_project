"""UserKV payload 序列化：tensor <-> 连续字节 blob。

遵循设计文档 §2.3 的内存布局：header（JSON：dtype/shape/层数）+ 逐层紧密拼接的原始张量字节。
禁止 pickle 任意对象；仅写入 dtype/形状元信息 + 连续字节。
"""

from __future__ import annotations

import json
import struct

import torch

_MAGIC = b"ONETRANSKV\x01"


def serialize(per_layer: list[tuple[torch.Tensor, torch.Tensor]]) -> bytes:
    """把逐层 (K_s^l, V_s^l) 序列化为单个连续字节 blob。

    每个张量按 ``[B, S, H, d]``（bshd）连续存储，取 ``.contiguous().numpy().tobytes()``。
    """
    header: dict = {"dtype": None, "n_layers": len(per_layer), "layers": []}
    body = bytearray()

    for l, (k, v) in enumerate(per_layer):
        k = k.detach().to("cpu").contiguous()
        v = v.detach().to("cpu").contiguous()
        assert k.dtype == v.dtype, "每层 K/V dtype 必须一致"
        header["layers"].append(
            {
                "l": l,
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "dtype": str(k.dtype).replace("torch.", ""),
            }
        )
        header["dtype"] = str(k.dtype).replace("torch.", "")
        body += k.numpy().tobytes()
        body += v.numpy().tobytes()

    header_bytes = json.dumps(header).encode("utf-8")
    return _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(body)


def deserialize(payload: bytes) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """把字节 blob 还原为逐层 (K_s^l, V_s^l)。"""
    if payload[: len(_MAGIC)] != _MAGIC:
        raise ValueError("未知 payload 魔数")
    pos = len(_MAGIC)
    (hlen,) = struct.unpack("<I", payload[pos : pos + 4])
    pos += 4
    header = json.loads(payload[pos : pos + hlen].decode("utf-8"))
    pos += hlen

    dtype = getattr(torch, header["dtype"])
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for meta in header["layers"]:
        k = _read_tensor(payload, pos, meta["k_shape"], dtype)
        pos += _nbytes(meta["k_shape"], dtype)
        v = _read_tensor(payload, pos, meta["v_shape"], dtype)
        pos += _nbytes(meta["v_shape"], dtype)
        out.append((k, v))
    return out


def _nbytes(shape: list[int], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * dtype.itemsize


def _read_tensor(payload: bytes, pos: int, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    nbytes = _nbytes(shape, dtype)
    buf = payload[pos : pos + nbytes]
    t = torch.frombuffer(bytearray(buf), dtype=dtype)
    return t.reshape(shape)