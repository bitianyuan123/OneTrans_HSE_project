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


def test_zero_copy() -> None:
    """KV 零拷贝数据面：deserialize 视图底层缓冲 + mmap 后端读侧零拷贝。"""
    import ctypes
    import tempfile

    from onetrans.serving.local_adapter import LocalKVStore
    from onetrans.serving.kv_store import UserKVRecord

    device = torch.device("cpu")
    runner = _build_runner(seed=3)
    s_emb, s_mask = _make_sequence(1, 29, device)
    kv = runner.encode_s(s_emb, s_mask)
    payload = serialize(kv.per_layer)

    # 1) frombuffer 零拷贝：deserialize 返回的张量直接视图底层 bytearray（无副本）
    buf = bytearray(payload)
    restored = deserialize(buf)
    carr = (ctypes.c_char * len(buf)).from_buffer(buf)
    base, end = ctypes.addressof(carr), ctypes.addressof(carr) + len(buf)
    ptr = restored[0][0].untyped_storage().data_ptr()
    assert base <= ptr < end, f"零拷贝失败：张量未视图底层缓冲 ptr={ptr} base={base}"
    for (k1, v1), (k2, v2) in zip(kv.per_layer, restored):
        assert (k1 - k2).abs().max().item() < 1e-6
        assert (v1 - v2).abs().max().item() < 1e-6

    # 2) mmap 后端：payload 映射为 memoryview，get 返回共享内存视图
    with tempfile.TemporaryDirectory() as mmap_dir:
        store = LocalKVStore(dtype="float32", mmap_dir=mmap_dir)
        key = KVKey(model_version="mv1", user_id="user1")
        store.put(UserKVRecord(
            key=key, s_len=kv.s_len, per_layer_len=kv.per_layer_len,
            dtype="float32", payload=payload,
        ))
        rec = store.get(key)
        assert rec is not None and isinstance(rec.payload, memoryview)
        restored_mm = deserialize(rec.payload)
        for (k1, v1), (k2, v2) in zip(kv.per_layer, restored_mm):
            assert (k1 - k2).abs().max().item() < 1e-6
            assert (v1 - v2).abs().max().item() < 1e-6
    print("KV 零拷贝： [ok] frombuffer 视图底层缓冲 + mmap 后端读侧零拷贝一致")


def test_routing_sharding() -> None:
    """一致性哈希路由 + 元数据失效 + 分片 KV 门面（数据本地化）。"""
    import time

    from onetrans.serving.kv_store import UserKVRecord
    from onetrans.serving.meta_store import KVPointer, LocalMetaStore, validate_pointer
    from onetrans.serving.router import JumpConsistentHash, RingHash, Router, hash64
    from onetrans.serving.sharded import ShardedKVStore

    device = torch.device("cpu")
    H, d = N_HEADS, D_MODEL // N_HEADS

    # 1) jump 一致性哈希：同一 key 稳定路由，桶数变化时 remap 比例受控
    jump = JumpConsistentHash(8)
    owners = [jump.shard_of(f"user-{i}") for i in range(1000)]
    assert all(0 <= o < 8 for o in owners)
    # 扩缩容到 9 桶：remap 应接近 1/9（理论下界），远小于全量迁移
    jump9 = JumpConsistentHash(9)
    owners9 = [jump9.shard_of(f"user-{i}") for i in range(1000)]
    from onetrans.serving.router import remap_ratio
    remap = remap_ratio(owners, owners9)
    assert remap < 0.2, f"jump hash remap 过高 {remap:.3f}"
    print(f"  一致性哈希(jump)： [ok] 8→9 桶 remap={remap:.3f}（<理论全量）")

    # 2) 环 hash：动态增删节点
    ring = RingHash(vnodes_per_node=32)
    ring.add_node("node-a")
    ring.add_node("node-b")
    assert ring.shard_of("user-x") in {"node-a", "node-b"}
    ring.remove_node("node-a")
    assert ring.shard_of("user-x") == "node-b"
    print("  一致性哈希(ring)： [ok] add/remove node 路由稳定")

    # 3) 稳定 hash：跨进程可复现（非 Python hash 随机盐）
    assert hash64("user-42") == hash64("user-42")

    # 4) 元数据指针校验：checksum 不一致触发降级判定
    meta = LocalMetaStore()
    ptr = KVPointer(
        model_version="mv1", user_id="u1", checksum="deadbeef",
        s_len=10, per_layer_len=[10], seq_ts_last=int(time.time()),
    )
    meta.set(ptr)
    assert meta.get("mv1", "u1") is not None
    good = UserKVRecord(
        key=KVKey("mv1", "u1"), s_len=10, per_layer_len=[10],
        dtype="float32", payload=b"abc",
    )
    assert validate_pointer(good, ptr) is False  # checksum 不匹配
    ptr2 = KVPointer(model_version="mv1", user_id="u1", checksum=good.checksum,
                     s_len=10, per_layer_len=[10], seq_ts_last=int(time.time()))
    assert validate_pointer(good, ptr2) is True
    meta.ttl("mv1", "u1", ttl_seconds=-1)  # 立即过期
    assert meta.get("mv1", "u1") is None  # 惰性过期
    print("  元数据失效： [ok] pointer 校验 + TTL 惰性过期")

    # 5) 分片 KV 门面：同一 user 恒落到同一 shard（数据本地化）
    shards = [
        build_kv_store(KVConfig(backend="local", dtype="float32"))
        for _ in range(3)
    ]
    sharded = ShardedKVStore(shards, router=Router(num_shards=3))
    uid = "user-shard"
    key = KVKey(model_version="mv1", user_id=uid)
    rec = UserKVRecord(
        key=key, s_len=10, per_layer_len=[10], dtype="float32",
        payload=serialize([(torch.randn(1, 10, H, d), torch.randn(1, 10, H, d))]),
    )
    sharded.put(rec)
    s0 = sharded.shard_of(uid)
    assert s0 == sharded.router.route(uid)  # 稳定
    got = sharded.get(key)
    assert got is not None and got.checksum == rec.checksum
    assert sharded.stores[s0].size() == 1  # 落在单一 shard，其余为空
    assert sum(s.size() for s in sharded.stores) == 1
    print("  分片 KV 门面： [ok] 同一 user 本地化命中，跨 shard 零复制")


def test_dynamic_batching() -> None:
    """动态 batching：攒批调度 + 批量 mget + score_ns_batch 数值等价单请求打分。"""
    from onetrans.serving.pipeline import BatchScheduler, ScoreRequest

    device = torch.device("cpu")
    runner = _build_runner(seed=4)
    store = build_kv_store(KVConfig(backend="local", dtype="float32"))
    nearline = NearlineWorker(runner, store, metrics=ServingMetrics(), dtype="float32")
    online = OnlineWorker(runner, store, metrics=ServingMetrics())

    # 三个不同历史长度用户，各 1 候选
    specs = [("u-b1", 17), ("u-b2", 33), ("u-b3", 45)]
    s_embs, s_masks, kv_refs = [], [], []
    for uid, valid in specs:
        s, m = _make_sequence(1, valid, device)
        nearline.ingest(s, m, user_id=uid, model_version="mv1")
        s_embs.append(s)
        s_masks.append(m)
        kv_refs.append(runner.encode_s(s, m))

    ns_refs = [torch.randn(1, NS_TOKENS, D_MODEL, device=device) for _ in specs]

    # 1) 批量打分路径：一次 mget + 一次 score_ns_batch
    batch = [
        ScoreRequest(key=KVKey("mv1", uid), ns_emb=ns) for (uid, _), ns in zip(specs, ns_refs)
    ]
    got = online.score_batch(batch)  # [3, 2]

    # 2) 参考：逐用户 score（单前向等价已由 test_equivalence 覆盖）
    refs = []
    for (s, m), kv, ns in zip(zip(s_embs, s_masks), kv_refs, ns_refs):
        refs.append(runner.score_ns(kv, ns))  # [1, 2]
    ref = torch.cat(refs, dim=0)
    assert (got - ref).abs().max().item() < 1e-4, f"批量打分不等价 {(got - ref).abs().max()}"
    print(f"动态 batching： [ok] {len(batch)} 用户攒批，score_ns_batch 与逐条一致")

    # 3) 攒批调度器：满批/超时语义 + 线程安全
    sched = BatchScheduler(max_batch_size=2, max_wait_seconds=0.01)
    sched.submit(batch[0])
    sched.submit(batch[1])
    assert len(sched) == 2
    first = sched.next_batch()
    assert len(first) == 2  # 满批立即返回
    # 不足满批时，超时后按已攒 ≥1 条返回
    sched.submit(batch[2])
    second = sched.next_batch()
    assert len(second) == 1
    print(f"  攒批调度器： [ok] 满批返回 {len(first)}，超时兜底返回 {len(second)}")


def test_dispatcher() -> None:
    """计算面线程模型：Dispatcher + WorkerPool + req_seq 异步匹配 + 背压。"""
    import threading
    import time

    from onetrans.serving.dispatcher import Dispatcher, OverloadRejected, WorkerPool

    # 模拟「计算耗时随 payload 变化」→ 乱序完成
    def handler(req):
        time.sleep(req.payload)
        return req.req_seq  # 回显 req_seq，验证异步匹配到正确调用方

    pool = WorkerPool(num_workers=3, queue_capacity=64, handler=handler)
    disp = Dispatcher(pool, mode="hash", backpressure_timeout=None)
    pool.start()

    # 1) 并发提交：结果按 req_seq 一一对应（乱序完成下仍正确匹配）
    n = 30
    delays = [(i * 37) % 7 for i in range(n)]  # 0..6ms 伪随机
    futs = []
    barrier = threading.Barrier(3)

    def producer(seed):
        barrier.wait()
        for i in range(seed, n, 3):
            futs.append((i, disp.submit(user_id=f"user-{i}", payload=delays[i] / 1000.0)))

    threads = [threading.Thread(target=producer, args=(s,)) for s in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(futs) == n
    resp_pairs = [(f"user-{i}", fut.result(timeout=5.0)) for i, fut in futs]
    # 每个响应回显的 user_id 必须与提交一致（证明 Future 无串扰，乱序完成仍正确匹配）
    for uid, resp in resp_pairs:
        assert resp.user_id == uid, f"异步匹配错乱：期望 {uid} 拿到 {resp.user_id}"
    # req_seq 全局唯一且完整覆盖 1..n（证明 req_seq 分配/回收无重复）
    seqs = sorted(resp.req_seq for _, resp in resp_pairs)
    assert seqs == list(range(1, n + 1)), "req_seq 应唯一覆盖 1..n"
    print(f"计算面线程模型： [ok] {n} 请求并发完成，req_seq 异步匹配一致")

    # 2) 数据本地化：同一 user_id 恒落同一 worker
    w0 = pool.worker_for("user-locality")
    assert w0 == pool.worker_for("user-locality")
    print(f"  数据本地化： [ok] 同 user 稳定映射 worker={w0}")

    # 3) 背压：容量 2 的单队列不启动 worker（不被消费），提交超限即被拒绝
    disp2 = Dispatcher(
        WorkerPool(num_workers=1, queue_capacity=2, handler=handler),
        mode="hash", backpressure_timeout=None,
    )
    futs2 = [disp2.submit(user_id=f"u{i}", payload=0.0) for i in range(5)]
    rejected = sum(
        1 for f in futs2
        if f.done() and isinstance(f.exception(), OverloadRejected)
    )
    pending = sum(1 for f in futs2 if not f.done())
    assert pending == 2 and rejected == 3, f"背压语义不符 pending={pending} rejected={rejected}"
    print(f"  背压： [ok] 队列容量=2 → 缓存 {pending} 条，拒绝 {rejected} 条")

    pool.stop()
    disp2.close()


def test_embedding_ps() -> None:
    """独立 PS 数据面：分片表 + 客户端 lookup + miss seed 兜底 + 权重版本。"""
    from onetrans.serving.embedding_ps_client import EmbeddingPSClient, LocalEmbeddingPS

    dim = 16
    ps = LocalEmbeddingPS(num_shards=8, dim=dim, seed=0)
    ids = torch.tensor([10, 21, 32], dtype=torch.long)
    weights = torch.randn(3, dim)
    ps.set_many(table="mv1", ids=ids, weights=weights)
    v1 = ps.version("mv1")
    assert v1 == 3  # 每个 id 写入触发一次版本递增

    # 绑定到已写入的同一本地 PS：验证「写侧表 → 客户端读侧」命中一致
    client = EmbeddingPSClient("mv1", dim=dim, local=ps)
    # 命中 + 未命中混查
    query = torch.tensor([10, 999, 21], dtype=torch.long)
    got = client.lookup(query)
    assert got.shape == (3, dim)
    # 命中 id 与写入一致；未命中 id 用 seed 兜底（确定性）
    assert (got[0] - weights[0]).abs().max().item() < 1e-6
    assert (got[2] - weights[1]).abs().max().item() < 1e-6
    # 同一 miss id 两次查应一致（确定性的 seed 兜底）
    again = client.lookup(torch.tensor([999], dtype=torch.long))
    assert (got[1] - again[0]).abs().max().item() < 1e-6
    assert got[1].abs().sum().item() > 0  # 兜底非全零
    print(f"独立 PS 数据面： [ok] 分片查表命中/seed 兜底确定性，版本={v1}")

    # 分片稳定性：同 feat_id 恒落同一分片
    from onetrans.serving.embedding_ps_client import ShardedEmbeddingTable
    tbl = ShardedEmbeddingTable(num_shards=8, dim=dim)
    assert tbl.shard_of(1234) == tbl.shard_of(1234)
    print(f"  分片稳定性： [ok] 同 id 稳定映射分片={tbl.shard_of(1234)}")


def test_weight_loader() -> None:
    import tempfile

    from onetrans.serving.weight_loader import load_backbone, save_checkpoint

    build_kwargs = dict(
        d_model=D_MODEL,
        num_blocks=N_BLOCKS,
        num_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        min_seq_len=MIN_SEQ_LEN,
        ns_tokens_num=NS_TOKENS,
        dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # 1) checkpoint 命中：保存 seed=7 权重 → 加载应逐位一致，source=checkpoint
        torch.manual_seed(7)
        ref = OneTrans(**build_kwargs).eval()
        save_checkpoint(ref, f"{ckpt_dir}/mv1.pt", model_version="mv1", seed=7)

        torch.manual_seed(1234)  # 干扰：加载前重置 RNG，证明权重来自 checkpoint 而非扰动
        loaded, src = load_backbone("mv1", checkpoint_dir=ckpt_dir, seed=1234, **build_kwargs)
        assert src == "checkpoint"
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), loaded.named_parameters()):
            assert n1 == n2 and (p1 - p2).abs().max().item() < 1e-7

        # 2) checkpoint 缺失 → seed 兜底，且与显式 seed 构建一致
        torch.manual_seed(42)
        expected = OneTrans(**build_kwargs).eval()
        torch.manual_seed(42)
        loaded2, src2 = load_backbone("mv_missing", checkpoint_dir=ckpt_dir, seed=42, **build_kwargs)
        assert src2 == "seed"
        for p1, p2 in zip(expected.parameters(), loaded2.parameters()):
            assert (p1 - p2).abs().max().item() < 1e-7

        # 3) checkpoint 损坏/结构不匹配 → seed 兜底（不抛异常）
        with open(f"{ckpt_dir}/bad.pt", "wb") as f:
            f.write(b"\x00\x01 not a torch checkpoint")
        loaded3, src3 = load_backbone("bad", checkpoint_dir=ckpt_dir, seed=0, **build_kwargs)
        assert src3 == "seed"

    print(f"权重版本化加载： [ok] checkpoint 命中一致 / 缺失与损坏均 seed 兜底")


def main() -> None:
    test_equivalence()
    test_serialize_roundtrip()
    test_tokenizer_split()
    test_append_conflict(lambda: build_kv_store(KVConfig(backend="local")))
    test_pipeline()
    test_zero_copy()
    test_routing_sharding()
    test_dynamic_batching()
    test_dispatcher()
    test_embedding_ps()
    test_weight_loader()
    print("\n全部端到端校验通过 ✅")


if __name__ == "__main__":
    main()