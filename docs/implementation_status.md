# 实现 & 现状总结

> 版本：v0.1
> 分支：`feat/onetrans-e2e-serving`
> 对应用户诉求：梳理「工程级详细设计」与「实现 & 现状」两份文档，并同步解决遗留的 P0/P1 问题。

---

## 1. 现状总览（一页）

序列 Transformer 精排（OneTrans 类）的**单机参照实现**已完成，覆盖从「行为流 → 近线 S 侧编码 → UserKV 存储/读取 → 在线 NS 交叉打分」的全链路，并通过 `demo.py` 的数值等价性 + 零拷贝 + 并发 + 路由 + 攒批 + 权重版本化 + PS 数据面等端到端校验。

生产侧（C++ brpc / datasystem / 稀疏 PS）以「接口契约 + 参考实现」形式给出：`deploy/ps/`（brpc 分片 PS）、`datasystem_adapter.py`（存储无关 adapter）、`engineering_design.md`（工程级方案）。未接入项见 §5。

---

## 2. 已完成功能（按功能点 / 提交粒度）

| # | 功能点 | 关键文件 | 提交 | 校验 |
|---|---|---|---|---|
| 1 | 修复 pyramid 剪尾方向（P0-1） | `nn/blocks/onetrans_block.py`、`serving/two_stage.py` | `2885cd2` | 逐位等价（max\|diff\|≈3e-8） |
| 2 | 权重版本化加载 + seed 兜底（P0-2） | `serving/weight_loader.py`、`run/train.py` | `229342d` | checkpoint 命中/缺失/损坏三态 |
| 3 | KV 零拷贝数据面 | `serving/serialize.py`、`serving/local_adapter.py` | `36513a9` | frombuffer 视图 + mmap 一致 |
| 4 | 一致性哈希路由 + 元数据/版本失效 + 分片 KV | `router.py`、`meta_store.py`、`sharded.py` | `05c4cdd` | remap≈0.116 / TTL / 本地化 |
| 5 | 动态 batching（攒批打分） | `pipeline.py`、`two_stage.py` | `ae2ccc6` | 攒批 vs 逐条等价 |
| 6 | 计算面线程模型（P1） | `dispatcher.py` | `4ed2436` | 并发完成 / 本地化 / 背压 |
| 7 | 独立稀疏参数服务器 PS | `deploy/ps/*`、`embedding_ps_client.py` | 本次提交 | 命中/seed 兜底/版本 |

---

## 3. 数值 / 等价性校验结果（demo.py 输出摘录）

```
等价性校验：
  valid_len=50 candidates=1  max|diff|=2.980e-08
  valid_len=23 candidates=1  max|diff|=0.000e+00
  valid_len=37 candidates=5  max|diff|=6.706e-08
KV 零拷贝： frombuffer 视图底层缓冲 + mmap 后端读侧零拷贝一致
一致性哈希(jump)： 8→9 桶 remap=0.116（<理论全量）
元数据失效： pointer 校验 + TTL 惰性过期
动态 batching： 3 用户攒批，score_ns_batch 与逐条一致
计算面线程模型： 30 请求并发完成，req_seq 异步匹配一致；背压拒绝 3 条
独立 PS 数据面： 分片查表命中/seed 兜底确定性，版本=3
权重版本化加载： checkpoint 命中一致 / 缺失与损坏均 seed 兜底
```

---

## 4. P0 / P1 问题清单与状态

| 级别 | 问题 | 状态 | 说明 |
|---|---|---|---|
| P0-1 | pyramid 方向错误（保留头而非尾） | ✅ 已修复 | 全链路改尾部保留 + 左 padding |
| P0-2 | 权重版本化加载（checkpoint + seed 兜底） | ✅ 已修复 | 缺失/损坏均回退 seed |
| P1 | 计算面线程模型（消全局锁 + req_seq 异步匹配 + 背压） | ✅ 已实现 | demo 并发/背压校验通过 |
| P1 | 独立稀疏参数服务器（brpc 分片 + 版本化） | ✅ 已实现 | 本次提交 |
| **P1** | **PS 客户端绑定错误（client 读取到空表）** | ✅ **本次修复** | demo `test_embedding_ps` 原将客户端绑定到新建空 PS，改为绑定已写入的 PS 实例 |

> 第 5 项为本次同步修复的遗留 P1：`EmbeddingPSClient.local()` 每次新建 `LocalEmbeddingPS`，导致「写侧表 → 客户端读侧」不一致，断言失败。已改为 `EmbeddingPSClient("mv1", dim=dim, local=ps)` 直接绑定同一实例。

---

## 5. 遗留差距（生产验收项，非正确性阻断）

| 项 | 级别 | 现状 | 备注 |
|---|---|---|---|
| PS remote 数据面（Python→brpc） | P1（验收） | `embedding_ps_client.py` `remote` 路径占位 `NotImplementedError` | 生产 brpc 调用在 C++ worker 内 |
| Redis 元数据后端 | P2 | `kv_store.py` `redis` 分支占位 | 可由 datasystem 元数据能力替代 |
| datasystem HBM 直通 | P2（环境） | `datasystem_adapter.py` 占位 | 需昇腾 NPU 联调 |
| vLLM 自定义 op 移植 | P1（验收） | 未开始 | 以 demo 数值基准为门槛 |

---

## 6. 提交策略

- **粒度**：按「修改点 / 功能」独立提交，commit message 用中文「类型: 描述」前缀（`fix:` / `feat:` / `perf:`）。
- **频率**：每次完成一个功能点或修复点即提交并推送到远端 `origin`。
- **文档**：`docs/` 变更随对应功能同批或独立提交，保持一致。