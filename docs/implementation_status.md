# 实现 & 现状总结

> 版本：v0.2（二次修订）
> 分支：`feat/onetrans-e2e-serving`
> 对应用户诉求：梳理「工程级详细设计」与「实现 & 现状」两份文档，评估与工程级推荐系统精排的差距。

---

## 1. 现状总览（一页）

序列 Transformer 精排（OneTrans 类）的**单机参照实现**已完成，覆盖「行为流 → 近线 S 侧编码 → UserKV 存储/读取 → 在线 NS 交叉打分」全链路，`demo.py` 通过数值等价性、零拷贝、并发、路由、攒批、权重版本化、PS 数据面等端到端校验。

生产侧（C++ brpc / datasystem / 稀疏 PS）以「接口契约 + 参考实现」形式给出：`deploy/ps/`（brpc 分片 PS）、`datasystem_adapter.py`（存储无关 adapter）、`engineering_design.md`（工程级方案）。**但距离工程级可用仍有明显差距**，主要集中在正确性细节、可靠性与可观测性（见 §5）。

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
| 7 | 独立稀疏参数服务器 PS | `deploy/ps/*`、`embedding_ps_client.py` | `68d4b98` | 命中/seed 兜底/版本 |

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
| P1 | 独立稀疏参数服务器（brpc 分片 + 版本化） | ✅ 已实现 | commit `68d4b98` |
| P1 | PS 客户端绑定错误（client 读到空表） | ✅ 已修复 | `test_embedding_ps` 改为绑定已写入的 `ps` 实例 |

---

## 5. 工程级差距评估（二次审阅产出）

对照「工程级可用的推荐系统精排」逐项评估，结论分三类：**正确性（P0/P1）、可靠性（P1）、工程化（P1/P2）**。

### 5.1 正确性（需修复，否则在线结果可能错误）

| 级别 | 问题 | 证据（代码位置） | 建议 |
|---|---|---|---|
| **P1** | datasystem 后端**丢失有效长度元数据**：payload 只序列化 dtype+shape，`YuanrongKVStore.put` 只写 `payload`、`get` 用全宽 shape 重建 `s_len`/`per_layer_len`，左 padding（用户历史短于 max_seq_len）时注意力有效掩码错误 | `serialize.py`（header 无 `s_len`/`per_layer_len`）、`datasystem_adapter.py` L56-77/L99-107 | `serialize` header 纳入 `s_len`+`per_layer_len`，或 datasystem `get` 从 `KVPointer` 取回并校验 checksum |
| **P1** | **PS 跨语言分片哈希不等价**：Python `hash64(str(id))`（sha256）≠ C++ `(id*Knuth)%n`；`embedding_server.cc` 注释误称「同构」 | `embedding_ps_client.py` L40-41 vs `embedding_server.cc` L33-37 | 统一到 C++ Knuth 乘法哈希（Python 改 `shard_of`），并对负 id 语义对齐 |
| **P1** | **C++ PS 仅单表**：忽略 `req.table()`，无法多模型版本/灰度 | `embedding_server.cc` L131（单 `ShardedEmbeddingTable`）、L105-120 | server 侧 `table→ShardedEmbeddingTable` 映射 + 版本 |

> 说明：本地后端（`LocalKVStore`）因「record 对象内联 `s_len`/`per_layer_len`」而正确，这属于**隐性依赖**，未固化到序列化契约——一旦切 datasystem 后端即触发 5.1 第一项回归，是「先单卡跑通、再切 datasystem」路径上的**最大的隐藏正确性风险**。

### 5.2 可靠性（生产必补，当前缺失）

| 能力 | 现状 | 建议 |
|---|---|---|
| 客户端超时 | `Future` 无 deadline | KV/PS/RPC 加超时与 cancel |
| 重试 & 幂等 | 无 | 读幂等重试；写幂等键/版本 |
| 熔断/限流 | 仅队列背压 | 按错误率熔断 + 令牌桶限流 |
| 健康检查 | 无 | `/healthz` + 依赖探针 |
| 优雅停机/排空 | `stop()` 仅 join | drain & wait 语义 |
| append 原子性 | `offset` 乐观校验（进程内） | datasystem 原子 CAS / fencing token（§5.1 之外的一致性项） |

### 5.3 可观测性（无法线上排障）

- 指标：`ServingMetrics` 仅内存收集，无 Prometheus/OTel 导出；percentile 全样本存储（O(n)）。
- 日志：无结构化日志（缺 req_id/trace_id 贯穿）。
- 追踪：无分布式 trace（路由/读 KV/打分/PS 查表）。

### 5.4 工程化 / 性能（P2）

- **无测试框架/CI**：`demo.py` 单脚本 `assert`，无 pytest/coverage/CI，回归保障弱。
- **局部性能**：`_project_ns`/`_apply_ns_ffn` 逐 token Python 循环（Ns=8 可容忍）；`RingHash` 建环 O(vnodes·n²)；percentile 全样本排序。
- **未接入项**：PS remote 数据面（Python→brpc）、redis 后端、datasystem HBM 直通、vLLM 自定义 op 移植。

---

## 6. 提交策略

- **粒度**：按「修改点 / 功能」独立提交，commit message 用中文「类型: 描述」前缀（`fix:` / `feat:` / `perf:` / `docs:`）。
- **频率**：每个功能点/修复点完成即提交并推送到远端 `origin`（凭据已配置，`credential.helper store`）。
- **文档**：`docs/` 变更随对应功能同批或独立提交。