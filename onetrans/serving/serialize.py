"""UserKV payload 序列化（零拷贝数据面）：tensor <-> 连续字节 blob。

遵循设计文档 §2.3 的内存布局：

    <magic:12B> <header_len:4B> <header_json> <raw_bytes>
    raw_bytes = concat(K_s^0, V_s^0, K_s^1, V_s^1, ..., K_s^{L-1}, V_s^{L-1})  # 均 bshd 连续

零拷贝要点（P1：消除 UserKV 数据面的冗余拷贝）：

- **读侧**：``deserialize`` 用 ``torch.frombuffer(buf, offset=...)`` 直接视图底层字节
  （``bytes`` / ``bytearray`` / ``memoryview`` / ``mmap``），不再 ``bytearray(buf)`` 切出副本。
  对 mmap 后端，反序列化的 K/V 张量直接指向共享内存，无 CPU RAM 二次拷贝。
- **写侧**：``serialize`` 预分配单一 ``bytearray``，用 ``ctypes.memmove`` 从张量
  ``data_ptr`` 一次拷贝到底层缓冲，避免逐层 ``+`` 拼接的 O(n²) 与中间 ``bytes`` 对象。

禁止 pickle 任意对象；仅写 dtype/形状元信息 + 连续字节。
"""

from __future__ import annotations

import ctypes
import json
import struct
from typing import Iterable, Sequence

import torch
from torch import Tensor

_MAGIC = b"ONETRANSKV\x01"
_HEADER_FMT = "<I"


def _nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * dtype.itemsize


def serialize(
    per_layer: Sequence[tuple[Tensor, Tensor]],
    s_len: int | None = None,
    per_layer_len: Sequence[int] | None = None,
) -> bytes:
    """把逐层 ``(K_s^l, V_s^l)`` 序列化为单个连续字节 blob。

    预分配总字节数，一次 memmove 拷贝每张量底层字节，避免重复分配。

    ``s_len``/``per_layer_len`` 为**有效长度元数据**（左 padding 语义下 ``K_s`` 的
    shape[1] 是 pyramid 该层满宽 ``dims[l]``，需显式区分有效 token 数，见
    ``docs/gap_analysis.md`` G1）。不传时回退为满宽（兼容无 padding 场景与旧调用）。
    """
    if per_layer_len is None:
        per_layer_len = [k.shape[1] for k, _ in per_layer]
    s_len = s_len if s_len is not None else (per_layer_len[0] if per_layer_len else 0)

    flat: list[Tensor] = []
    meta: list[tuple[int, str]] = []  # (element_size, dtype_str)
    for k, v in per_layer:
        k = k.detach().to("cpu").contiguous()
        v = v.detach().to("cpu").contiguous()
        assert k.dtype == v.dtype, "每层 K/V dtype 必须一致"
        flat.append(k)
        flat.append(v)
        meta.append((k.element_size(), str(k.dtype).replace("torch.", "")))
        meta.append((v.element_size(), str(v.dtype).replace("torch.", "")))

    header = {
        "dtype": meta[0][1] if meta else "float32",
        "n_layers": len(per_layer),
        "s_len": s_len,
        "layers": [
            {"l": i, "k_shape": list(k.shape), "v_shape": list(v.shape), "len": per_layer_len[i]}
            for i, (k, v) in enumerate(per_layer)
        ],
    }
    header_bytes = json.dumps(header).encode("utf-8")

    body_bytes = sum(t.numel() * t.element_size() for t in flat)
    total = len(_MAGIC) + 4 + len(header_bytes) + body_bytes
    buf = bytearray(total)
    pos = 0
    buf[pos : pos + len(_MAGIC)] = _MAGIC
    pos += len(_MAGIC)
    struct.pack_into(_HEADER_FMT, buf, pos, len(header_bytes))
    pos += 4
    buf[pos : pos + len(header_bytes)] = header_bytes
    pos += len(header_bytes)

    # 单次 memmove 拷贝每张量底层字节到预分配缓冲
    carr = (ctypes.c_char * len(buf)).from_buffer(buf)
    dst_base = ctypes.addressof(carr)
    for t in flat:
        n = t.numel() * t.element_size()
        ctypes.memmove(dst_base + pos, t.data_ptr(), n)
        pos += n

    return bytes(buf)


def _parse_header(payload: bytes | bytearray | memoryview) -> tuple[dict, int]:
    """解析并校验 ``<magic><header_len><header_json>``，返回 ``(header, 其后偏移)``。"""
    if bytes(payload[: len(_MAGIC)]) != _MAGIC:
        raise ValueError("未知 payload 魔数")
    pos = len(_MAGIC)
    (hlen,) = struct.unpack_from(_HEADER_FMT, payload, pos)
    pos += 4
    header = json.loads(bytes(payload[pos : pos + hlen]).decode("utf-8"))
    pos += hlen
    return header, pos


def read_header(payload: bytes | bytearray | memoryview) -> dict:
    """仅解析 header，返回 ``dict``（含 ``s_len`` 与各层 ``layers[].len``）。

    供读侧只需有效长度元数据、而不反序列化整对象张量的场景（如 datasystem
    ``get``/``append``，见 ``docs/gap_analysis.md`` G1）。
    """
    header, _ = _parse_header(payload)
    return header


def deserialize_with_meta(
    payload: bytes | bytearray | memoryview,
) -> tuple[list[tuple[Tensor, Tensor]], int, list[int]]:
    """反序列化并返回 ``(per_layer, s_len, per_layer_len)``。

    有效长度来自 header（``s_len`` / ``layers[].len``）；旧数据（无这些字段）回退为
    「满宽有效」，与旧语义一致（向后兼容，见 ``docs/gap_analysis.md`` G1）。
    """
    header, _ = _parse_header(payload)
    per_layer = deserialize(payload)
    s_len = header.get("s_len")
    if s_len is None:
        s_len = per_layer[0][0].shape[1] if per_layer else 0
    per_layer_len = [m.get("len", m["k_shape"][1]) for m in header["layers"]]
    return per_layer, s_len, per_layer_len


def deserialize(payload: bytes | bytearray | memoryview) -> list[tuple[Tensor, Tensor]]:
    """把字节 blob 还原为逐层 ``(K_s^l, V_s^l)``（零拷贝视图）。

    :param payload: 底层缓冲；``bytes`` 得到只读视图，``bytearray``/``memoryview``/``mmap``
        得到可直接写回的视图（零拷贝）。返回张量与底层缓冲共享存储。
    """
    header, pos = _parse_header(payload)
    dtype = getattr(torch, header["dtype"])
    out: list[tuple[Tensor, Tensor]] = []
    for meta in header["layers"]:
        k = _view_tensor(payload, pos, meta["k_shape"], dtype)
        pos += _nbytes(meta["k_shape"], dtype)
        v = _view_tensor(payload, pos, meta["v_shape"], dtype)
        pos += _nbytes(meta["v_shape"], dtype)
        out.append((k, v))
    return out


def _view_tensor(
    payload: bytes | bytearray | memoryview,
    pos: int,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> Tensor:
    numel = 1
    for s in shape:
        numel *= s
    # frombuffer 直接视图底层缓冲（offset 指向各层偏移），无需副本
    return torch.frombuffer(payload, dtype=dtype, count=numel, offset=pos).reshape(shape)


def per_layer_offsets(payload: bytes | bytearray | memoryview) -> list[tuple[int, int]]:
    """返回每层 ``(K 偏移, V 偏移)``（用于按层抽取而不反序列化整对象，见设计文档 §2.3）。"""
    header, pos = _parse_header(payload)
    dtype = getattr(torch, header["dtype"])
    offsets: list[tuple[int, int]] = []
    for meta in header["layers"]:
        k_pos = pos
        pos += _nbytes(meta["k_shape"], dtype)
        v_pos = pos
        pos += _nbytes(meta["v_shape"], dtype)
        offsets.append((k_pos, v_pos))
    return offsets


def iterate_tensors(
    payload: bytes | bytearray | memoryview,
) -> Iterable[Tensor]:
    """惰性迭代全部逐层张量（零拷贝视图），供逐层处理用。"""
    yield from (t for pair in deserialize(payload) for t in pair)