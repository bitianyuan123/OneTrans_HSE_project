"""yuanrong datasystem 后端 adapter（懒加载）。

把 :class:`KVStore` 接口映射到 openYuanrong datasystem 的 KV 语义：
- DRAM 驻留/持久化缓存 -> ``kv().set / get([key])``（共享内存免拷贝）
- HBM 直通/跨卡投放 -> ``hetero().DevPublish / DevSubscribe``（可选，见 ``prefetch``）
- 冷层持久化 -> ``writeMode=WRITE_THROUGH_L2_CACHE``（OBS/SFS，可选）

注意：本模块 import 即惰性，SDK 缺失时不 crash（仅在使用时报错），
以便在无 yuanrong 环境（如本仓库 CI）下仍可运行其余端到端路径。

对应设计文档 §1.4 / §1.5。
"""

from __future__ import annotations

from typing import Any, Optional

from onetrans.serving.kv_store import (
    AppendResult,
    DeleteResult,
    DeltaKV,
    KVKey,
    KVStore,
    PutResult,
    UserKVRecord,
)


class YuanrongKVStore(KVStore):
    def __init__(self, host: str = "127.0.0.1", port: int = 31501, dtype: str = "float16", **_: Any) -> None:
        self._host = host
        self._port = port
        self._dtype = dtype
        self._client = None

    def _require(self) -> Any:
        if self._client is None:
            try:
                from yr.datasystem.ds_client import DsClient  # type: ignore
            except ImportError as e:  # pragma: no cover - SDK 未安装
                raise RuntimeError(
                    "yuanrong datasystem SDK 未安装；请先部署 datasystem 或改用 lower backend"
                ) from e
            self._client = DsClient(self._host, self._port)
            self._client.init()
        return self._client

    def connect(self, conf: dict[str, Any] | None = None) -> None:
        # 连接在首次使用时惰性建立；显式 connect 立即验证可达性
        self._require()
        return None

    def close(self) -> None:
        self._client = None

    def put(self, rec: UserKVRecord) -> PutResult:
        # writeMode 默认 none：UserKV 可由 nearline 幂等重放重建
        self._require().kv().set(str(rec.key), rec.payload)
        return PutResult(accepted=True, version=rec.key.model_version, checksum=rec.checksum)

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> Optional[UserKVRecord]:
        from onetrans.serving.serialize import deserialize_with_meta  # 局部 import，避免 SDK 侧依赖

        val = self._require().kv().get([str(key)])
        if not val or val[0] is None:
            return None
        payload: bytes = val[0]
        per_layer, s_len, per_layer_len = deserialize_with_meta(payload)
        if layers is not None:
            idx = [l for l in layers if 0 <= l < len(per_layer)]
            per_layer = [per_layer[l] for l in idx]
            per_layer_len = [per_layer_len[l] for l in idx]
        if not per_layer:
            return None
        return UserKVRecord(
            key=key,
            s_len=s_len,
            per_layer_len=per_layer_len,
            dtype=self._dtype,
            payload=_serialize_subset(per_layer, s_len=s_len, per_layer_len=per_layer_len),
        )

    def mget(self, keys: list[KVKey], *, layers: list[int] | None = None) -> list[Optional[UserKVRecord]]:
        return [self.get(k, layers=layers) for k in keys]

    def append(self, delta: DeltaKV) -> AppendResult:
        # NOTE: datasystem KV 无原生原子 append；用「读-合并-写」实现，
        # 依赖调用方（nearline）按 user 分区串行写，避免并发冲突。
        rec = self.get(delta.key)
        if rec is None:
            return AppendResult(False, 0, "", reason="missing")
        if delta.offset != rec.s_len:
            return AppendResult(False, rec.s_len, "", reason="offset_conflict")
        if delta.expect_checksum and delta.expect_checksum != rec.checksum:
            # G2 CAS fencing：expect_checksum 不匹配即拒绝，消除读-合并-写 TOCTOU 丢写
            return AppendResult(False, rec.s_len, "", reason="cas_conflict")
        import torch

        from onetrans.serving.serialize import deserialize

        per_layer = deserialize(rec.payload)
        merged = [
            (torch.cat([p[0], d[0]], dim=1), torch.cat([p[1], d[1]], dim=1))
            for p, d in zip(per_layer, delta.tensors)
        ]
        new_s_len = rec.s_len + delta.delta_len
        new_per_layer_len = [pl + delta.delta_len for pl in rec.per_layer_len]
        new_rec = UserKVRecord(
            key=delta.key,
            s_len=new_s_len,
            per_layer_len=new_per_layer_len,
            dtype=self._dtype,
            payload=_serialize_subset(merged, s_len=new_s_len, per_layer_len=new_per_layer_len),
            seq_ts_last=rec.seq_ts_last,
        )
        self._require().kv().set(str(delta.key), new_rec.payload)
        return AppendResult(True, new_rec.s_len, new_rec.checksum)

    def delete(self, keys: list[KVKey]) -> DeleteResult:
        self._require().kv().delete([str(k) for k in keys])
        return DeleteResult(deleted=len(keys))

    def ttl(self, key: KVKey, ttl_seconds: int) -> None:
        # datasystem 生命周期由写入参数控制；TTL 语义可经元数据面落地（TODO）
        raise NotImplementedError("TTL 需经 datasystem 写入参数/二级缓存策略配置")

    def prefetch(self, keys: list[KVKey], *, dest: str = "hbm") -> list[Any]:
        # HBM 直通走 heterogeneous object（DevPublish/DevSubscribe）。
        # 此处仅预留接口；需要昇腾 NPU 环境与 hetero 注册流程（见设计文档 §1.4）。
        raise NotImplementedError("异构对象 HBM 直通需昇腾 NPU 环境")


def _serialize_subset(
    per_layer: list[tuple[Any, Any]],
    s_len: int | None = None,
    per_layer_len: list[int] | None = None,
) -> bytes:
    from onetrans.serving.serialize import serialize

    return serialize(per_layer, s_len=s_len, per_layer_len=per_layer_len)