"""端到端演示与数值等价性校验（无数据管线/无 wandb/无 cuda/datasystem 依赖）。

运行：``python -m onetrans.serving.demo``

覆盖：
1. 单前向 vs 两阶段（encode_s + score_ns）数值等价（无 padding / 有 padding / 多候选广播）。
2. UserKV payload 序列化 roundtrip 一致性。
3. KV append 乐观并发（offset 冲突）校验。
4. Nearline/Online worker 经 LocalKVStore 的完整读写链路。
5. Tokenizer 的 encode_s / encode_ns 拆分正确性。
6. 指标快照输出。
"""

from __future__ import annotations

import torch

from onetrans.models.one_trans import OneTrans
from onetrans.nn.tokenizer import NSGroupWiseTokenizer, OneTransTokenizer, STokenizer
from onetrans.serving.kv_store import DeltaKV, KVConfig, KVKey, build_kv_store
from onetrans.serving.metrics import ServingMetrics, report_table
from onetrans.serving.pipeline import NearlineWorker, OnlineWorker
from onetrans.serving.serialize import deserialize, serialize
from onetrans.serving.two_stage import TwoStageRunner

# 与小配置对齐：d_model 可被 num_heads 整除
D_MODEL = 128
N_HEADS = 4
N_BLOCKS = 4
MAX_SEQ_LEN = 50
MIN_SEQ_LEN = 5
NS_TOKENS = 8


def _build_runner(seed: int = 0) -> TwoStageRunner:
    torch.manual_seed(seed)
    backbone = OneTrans(
        d_model=D_MODEL,
        num_blocks=N_BLOCKS,
        num_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        min_seq_len=MIN_SEQ_LEN,
        ns_tokens_num=NS_TOKENS,
        dropout=0.0,
    ).eval()
    return TwoStageRunner(backbone)


def _make_sequence(batch: int, valid_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 left-padding 的 S 序列嵌入与掩码（后 valid_len 有效，前部 padding 为零）。"""
    s = torch.randn(batch, MAX_SEQ_LEN, D_MODEL, device=device)
    mask = torch.zeros(batch, MAX_SEQ_LEN, dtype=torch.bool, device=device)
    mask[:, MAX_SEQ_LEN - valid_len :] = True
    s = s * mask.unsqueeze(-1)
    return s, mask


def _single_forward(runner: TwoStageRunner, s_emb: torch.Tensor, s_mask: torch.Tensor, ns_emb: torch.Tensor, ns_mask: torch.Tensor) -> torch.Tensor:
    tokens = torch.cat([s_emb, ns_emb], dim=1)
    mask = torch.cat([s_mask, ns_mask], dim=1)
    return runner.backbone(tokens, mask)


def test_equivalence() -> None:
    device = torch.device("cpu")
    runner = _build_runner()

    def check(valid_len: int, n_cand: int) -> None:
        s_emb, s_mask = _make_sequence(1, valid_len, device)  # B=1 用户历史
        ns_emb = torch.randn(n_cand, NS_TOKENS, D_MODEL, device=device)  # B=M 候选
        ns_mask = torch.ones(n_cand, NS_TOKENS, dtype=torch.bool, device=device)

        # 多候选单前向：同一用户历史广播到每个候选
        s_b = s_emb.expand(n_cand, -1, -1)
        s_m = s_mask.expand(n_cand, -1)
        ref = _single_forward(runner, s_b, s_m, ns_emb, ns_mask)  # [M, 2]

        # 两阶段：编码一次用户历史，再对全部候选打分
        kv = runner.encode_s(s_emb, s_mask)
        got = runner.score_ns(kv, ns_emb)  # [M, 2]
        diff = (ref - got).abs().max().item()
        assert diff < 1e-4, f"等价性失败 valid_len={valid_len} cand={n_cand} diff={diff:.3e}"
        print(f"  [ok] valid_len={valid_len:>2} candidates={n_cand}  max|diff|={diff:.3e}  "
              f"ref={ref[0].tolist()}")

    print("等价性校验：")
    check(valid_len=MAX_SEQ_LEN, n_cand=1)  # 满序列、单候选
    check(valid_len=23, n_cand=1)  # 有 padding、单候选
    check(valid_len=37, n_cand=5)  # 有 padding、多候选广播


def test_serialize_roundtrip() -> None:
    device = torch.device("cpu")
    runner = _build_runner(seed=1)
    s_emb, s_mask = _make_sequence(1, 31, device)
    kv = runner.encode_s(s_emb, s_mask)

    payload = serialize(kv.per_layer)
    restored = deserialize(payload)
    for (k1, v1), (k2, v2) in zip(kv.per_layer, restored):
        assert k1.shape == k2.shape and v1.shape == v2.shape
        assert (k1 - k2).abs().max().item() < 1e-6
        assert (v1 - v2).abs().max().item() < 1e-6
    print(f"序列化 roundtrip： [ok] {len(payload)} bytes, {len(kv.per_layer)} 层逐层一致")


def test_append_conflict(build_kv_store_fn) -> None:
    import time

    from onetrans.serving.kv_store import UserKVRecord

    device = torch.device("cpu")
    H, d = N_HEADS, D_MODEL // N_HEADS
    base_k = torch.randn(1, 10, H, d, device=device)
    base_v = torch.randn(1, 10, H, d, device=device)
    dk = torch.randn(1, 3, H, d, device=device)
    dv = torch.randn(1, 3, H, d, device=device)

    store = build_kv_store_fn()
    key = KVKey(model_version="mv1", user_id="user1")
    rec = UserKVRecord(
        key=key,
        s_len=10,
        per_layer_len=[10],
        dtype="float32",
        payload=serialize([(base_k, base_v)]),
        created_at=int(time.time()),
    )
    store.put(rec)

    # 正确 offset
    ok = store.append(DeltaKV(key=key, base_version="mv1", offset=10, delta_len=3, tensors=[(dk, dv)]))
    # 冲突 offset
    bad = store.append(DeltaKV(key=key, base_version="mv1", offset=999, delta_len=3, tensors=[(dk, dv)]))
    assert ok.accepted is True and bad.accepted is False and bad.reason == "offset_conflict"
    print(f"append 乐观并发： [ok] 正确 offset 接受={ok.accepted}, 冲突拒绝={bad.accepted} ({bad.reason})")
    # offset 冲突时 new_s_len 应回传当前 s_len（首个 append 成功后为 13）
    assert bad.new_s_len == 13


def test_pipeline() -> None:
    device = torch.device("cpu")
    runner = _build_runner(seed=2)
    metrics = ServingMetrics()
    store = build_kv_store(KVConfig(backend="local", dtype="float32"))
    nearline = NearlineWorker(runner, store, metrics, dtype="float32")
    online = OnlineWorker(runner, store, metrics)

    uid, mv = "user-42", "onetrans@v1"
    s_emb, s_mask = _make_sequence(1, 25, device)
    res = nearline.ingest(s_emb, s_mask, user_id=uid, model_version=mv)
    assert res.accepted

    ns_emb = torch.randn(3, NS_TOKENS, D_MODEL, device=device)
    got = online.score(uid, mv, ns_emb)  # [3, 2]
    # 与直接两阶段一致
    kv = runner.encode_s(s_emb, s_mask)
    ref = runner.score_ns(kv, ns_emb)
    assert (ref - got).abs().max().item() < 1e-4
    print(f"pipeline 读写链路： [ok] put->get 命中, 3 候选打分一致")

    print(report_table(metrics.snapshot(), title="serving metrics"))
    return metrics


def test_tokenizer_split() -> None:
    d_model = D_MODEL
    s_tok = STokenizer(d_model, in_dims=[16], merge="timestamp_agnostic")
    ns_tok = NSGroupWiseTokenizer(d_model, in_dims=[8] * NS_TOKENS)
    tok = OneTransTokenizer(s_tok, ns_tok, d_model, MAX_SEQ_LEN, use_cls_token=False)

    seq_f = [torch.randn(1, MAX_SEQ_LEN, 16)]
    seq_m = [torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool)]
    ns_g = [torch.randn(1, 8) for _ in range(NS_TOKENS)]

    s_tokens, s_mask = tok.encode_s(seq_f, seq_m)
    ns_tokens, ns_mask = tok.encode_ns(ns_g)
    assert s_tokens.shape == (1, MAX_SEQ_LEN, d_model)
    assert ns_tokens.shape == (1, NS_TOKENS, d_model)

    full, full_mask = tok.forward({"seq_features": seq_f, "seq_masks": seq_m, "ns_groups": ns_g})
    cat_ref = torch.cat([s_tokens, ns_tokens], dim=1)
    assert (full - cat_ref).abs().max().item() < 1e-6
    print(f"tokenizer 拆分： [ok] encode_s {tuple(s_tokens.shape)} / encode_ns {tuple(ns_tokens.shape)} 与 forward 一致")


def main() -> None:
    test_equivalence()
    test_serialize_roundtrip()
    test_tokenizer_split()
    test_append_conflict(lambda: build_kv_store(KVConfig(backend="local")))
    test_pipeline()
    print("\n全部端到端校验通过 ✅")


if __name__ == "__main__":
    main()