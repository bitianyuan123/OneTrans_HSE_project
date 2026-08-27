# 实现 & 现状总结

> 版本：v1.0（C++ 工程主线确立；Python 单机参照降为对拍基准）
> 分支：`feat/onetrans-e2e-serving`
> 文档定位（三分体系 ③ 现状 & 差距）：本文记录**已实现/已验证**的落地状态与实测结果；差距分级与路线图见 [gap_analysis.md](./gap_analysis.md)；第二阶段执行设计见 [phase2_design.md](./phase2_design.md)；模型层见 [model_design.md](./model_design.md)；端到端设计见 [detailed_design.md](./detailed_design.md)。

---

## 0. C++ 工程主线（当前形态）

**仓库现状**：`cpp/` 下已存在完整的端到端 C++ 实现（`a0c3eb5`，40 文件 / +5787 行），是当前的工程主线。Python 单机参照（`onetrans/`、`demo.py`）保留为**数值对拍基准与教学参照**，不再承接工程演进。

### 0.1 已交付（C++ 端到端，零第三方依赖）

| 交付物 | 位置 | 验证 |
|---|---|---|
| `onetrans_core` 静态库 | [cpp/src/](file:///workspace/cpp/src)（common/engine/kv/serving/net） | 数值对拍 24/24 PASS |
| `onetrans_server` HTTP 服务 | [cpp/tools/server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) | e2e 7/7 PASS |
| `verify_golden` 对拍工具 | [cpp/tools/verify_golden.cpp](file:///workspace/cpp/tools/verify_golden.cpp) | max diff < 1e-6 |
| 权重/golden 导出 | [cpp/tools/export_weights.py](file:///workspace/cpp/tools/export_weights.py) + `gen_golden.py` | manifest + 二进制落盘 |
| 端到端测试 | [cpp/tools/e2e_test.py](file:///workspace/cpp/tools/e2e_test.py) | ingest→score→对拍 golden |

### 0.2 实测结果

```
verify_golden：共 24 项，失败 0 项（C++ vs Python 全链路逐位等价，max diff < 1e-6）
e2e_test（7/7 PASS）：
  healthz / ingest(accepted, s_len=37) / score(max diff 2.38e-07 vs golden)
  / kv_miss 降级全零 / metrics / route 404·405
```

### 0.3 线程模型现状 → 目标

当前 `cpp/` 已实现「**接入/计算分离 + 攒批**」：`net/http_server` accept 线程 + worker 池负责收发；`serving/pipeline` 的 `Dispatcher` + `BatchScheduler` 用独立 `score_threads` 消费批次做 `score_ns_batch`。**但 lookup/KV/encode 仍在 compute 线程内同步执行**（见 [detailed_design §7.4.9](./detailed_design.md)）。

下一阶段目标：引入 **folly 功能分池 + `folly::Future` 阶段串联**，把 I/O（embedding lookup / KV I/O）与 CPU（encode / score）彻底解耦，达到 §7.4 的目标线程模型：

| 阶段 | 目标池 | 当前落点 | 待办 |
|---|---|---|---|
| Ingress | brpc bthread（仅收发+序列化） | `net/http_server`（已分离 accept，但 `/score` 同步 `fut.get()`） | 改异步回调 |
| EmbedLookup | `IOThreadPoolExecutor` | `embed_lookup`（回调可注入，当前同步） | 迁出独立 I/O 池 |
| Frontend Encode | `CPUThreadPoolExecutor` | `engine/frontend`（与 compute 同线程） | 迁出独立池 |
| KV I/O | `IOThreadPoolExecutor` | `kv/store` LocalKVStore（当前同步） | 迁出独立 I/O 池 |
| Compute | `CPUThreadPoolExecutor`（绑核） | `pipeline` Dispatcher+BatchScheduler | **已分池** |
| 阶段串联 | folly `Future` + `via(executor)` | 无（同线程串行） | 引入串联 |

---

## 1. 现状总览（Python 参照基准，降级为对拍底座）

序列 Transformer 精排（OneTrans 类）的**单机参照实现**已固化，覆盖「行为流 → 近线 S 侧编码 → UserKV 存储/读取 → 在线 NS 交叉打分」全链路，`demo.py` 通过数值等价性、零拷贝、并发、路由、攒批、权重版本化、PS 数据面等端到端校验。**M5 正确性收口已完成**（G1 元数据固化 / G2 append CAS fencing / G3 路由统一 / G8 KV miss 降级）。该实现的职责现为 C++ 主线的**数值对拍基准**（见 §0）；C++ 主线的接入/数据面/编排见 [detailed_design.md §7.4](./detailed_design.md)。

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
| 8 | M5 正确性收口：G1 元数据固化 + G2 append CAS + G3 路由统一 + G8 miss 降级 | `serialize.py`、`kv_store.py`、`pipeline.py`、`local_adapter.py`、`datasystem_adapter.py`、`dispatcher.py` | （本次提交） | header 固化 roundtrip / cas_conflict 拒绝 / worker_for==Router.route / miss 全零 |

---

## 3. 数值 / 等价性校验结果（demo.py 输出摘录）

```
等价性校验：
  valid_len=50 candidates=1  max|diff|=2.980e-08
  valid_len=23 candidates=1  max|diff|=0.000e+00
  valid_len=37 candidates=5  max|diff|=6.706e-08
KV 零拷贝： frombuffer 视图底层缓冲 + mmap 后端读侧零拷贝一致
  有效长度元数据： s_len=31, per_layer_len=[31, 31, 27, 16] 随 header 固化（G1）
append 乐观并发： 正确 offset 接受 / offset_conflict 拒绝 / cas_conflict 拒绝（G2）
  KV miss 降级： 缺失用户返回全零且不抛异常（G8）
一致性哈希(jump)： 8→9 桶 remap=0.116（<理论全量）
元数据失效： pointer 校验 + TTL 惰性过期
动态 batching： 3 用户攒批，score_ns_batch 与逐条一致
  miss/hit 混查： 缺失行全零、命中行与逐条一致（G8）
计算面线程模型： 30 请求并发完成，req_seq 异步匹配一致；背压拒绝 3 条
  路由统一： worker_for == Router.route（jump 哈希）（G3）
独立 PS 数据面： 分片查表命中/seed 兜底确定性，版本=3
  分片稳定性： 同 id 稳定映射分片=2（Knuth 跨语言等价）
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
| P1 | G1 datasystem 元数据丢失 / G2 append CAS / G3 路由统一 / G8 miss 降级（M5 正确性收口） | ✅ 已实现 | 见 §2 第 8 行；`demo.py` 全部断言通过 |

---

## 5. 工程级差距评估（二次审阅产出）

对照「工程级可用的推荐系统精排」逐项评估，结论分三类：**正确性（P0/P1）、可靠性（P1）、工程化（P1/P2）**。

### 5.1 正确性（需修复，否则在线结果可能错误）

| 级别 | 问题 | 证据（代码位置） | 建议 |
|---|---|---|---|
| **P1** | ~~datasystem 后端**丢失有效长度元数据**~~ | ✅ **M5 已修复**：`serialize` header 固化 `s_len`+`per_layer_len`（`serialize.py` `read_header`/`deserialize_with_meta`，向后兼容），`NearlineWorker.ingest` 写入、`YuanrongKVStore.get`/`append` 读回，datasystem 后端与 local 语义一致 |
| **P1** | **PS 跨语言分片哈希不等价**：Python `hash64(str(id))`（sha256）≠ C++ `(id*Knuth)%n`；`embedding_server.cc` 注释误称「同构」 | ✅ **已修复**：统一到 C++ Knuth 乘法哈希（Python `shard_of` 改 `hash64`，负 id 语义对齐） |
| **P1** | **C++ PS 仅单表**：忽略 `req.table()`，无法多模型版本/灰度 | ✅ **已修复**：server 侧 `table→ShardedEmbeddingTable` 映射 + 版本，`DoLookup` 按 `req.table()` 路由 |
| **P1** | ~~路由哈希方案不统一（三次审阅新增）~~ | ✅ **M5 已修复**：`WorkerPool.worker_for` 从 `hash64%n` 取模改为复用同一 `Router`（jump 一致性哈希），worker 分派与 KV 分片同桶，扩缩容 remap 受控（8→9 桶 ≈0.116） |

> 说明：~~本地后端（`LocalKVStore`）因「record 对象内联 `s_len`/`per_layer_len`」而正确，这属于**隐性依赖**，未固化到序列化契约。~~ **已消除**：M5 起有效长度显式固化进 `serialize` header，datasystem 后端读写与 local 语义一致，隐性依赖转为显式契约。

### 5.2 可靠性（生产必补，当前缺失）

| 能力 | 现状 | 建议 |
|---|---|---|
| 客户端超时 | `Future` 无 deadline | KV/PS/RPC 加超时与 cancel |
| 重试 & 幂等 | 无 | 读幂等重试；写幂等键/版本 |
| 熔断/限流 | 仅队列背压 | 按错误率熔断 + 令牌桶限流 |
| 健康检查 | 无 | `/healthz` + 依赖探针 |
| 优雅停机/排空 | `stop()` 仅 join | drain & wait 语义 |
| append 原子性 | ✅ **M5 已实现**：`DeltaKV.expect_checksum`（fencing token）CAS，offset+checksum 双校验，`cas_conflict` 拒绝不丢写 | （datasystem 原生原子 CAS 仍可后置） |

### 5.3 可观测性（无法线上排障）

- 指标：`ServingMetrics` 仅内存收集，无 Prometheus/OTel 导出；percentile 全样本存储（O(n)）。
- 日志：无结构化日志（缺 req_id/trace_id 贯穿）。
- 追踪：无分布式 trace（路由/读 KV/打分/PS 查表）。

### 5.4 工程化 / 性能（P2）

- **无测试框架/CI**：`demo.py` 单脚本 `assert`，无 pytest/coverage/CI，回归保障弱。
- **局部性能**：`_project_ns`/`_apply_ns_ffn` 逐 token Python 循环（Ns=8 可容忍）；`RingHash` 建环 O(vnodes·n²)；percentile 全样本排序。
- **未接入项**：PS remote 数据面（Python→brpc）、redis 后端、datasystem HBM 直通、vLLM 自定义 op 移植。

### 5.5 集成 / 落地缺口（P1，三次审阅新增）

对照「工程级精排」的**端到端可运行性**，除 §5.1~§5.4 外仍有**尚未接线的链路**，属「单卡参照已通、集群落地未通」：

| 级别 | 缺口 | 证据 | 建议 |
|---|---|---|---|
| **P1** | 无 C++ Nearline/Online 热路径 worker：两阶段 brpc 分离部署目前仅 PS（`deploy/ps`）有 C++ 参考实现，混合参数化层未移植 vLLM 自定义 op | `deploy/ps/`（仅 PS）、`nn/attention/mixed_attention.py`、`nn/ffn/mixed_ffn.py` | 以 Python 参照为数值基准，移植 Stage I/II 到 brpc+bthread worker + vLLM 自定义层 |
| **P1** | tokenizer + 稀疏 embedding 查表未接入 serving 热路径：`ingest`/`score` 直接收已 tokenize 的 `s_emb`/`ns_emb`，行为流→查表→编码、特征服务→查表→打分未接线 | `pipeline.py`（`NearlineWorker.ingest` / `OnlineWorker.score` 签名收 `s_emb`/`ns_emb`） | 接通 `OneTransTokenizer` + `EmbeddingPSClient`（fabric ①）到 pipeline 入口 |
| **P1** | ~~KV miss 硬失败（`raise KeyError`），无「陈旧读+打点」或「空 KV 快速返回」~~ | ✅ **M5 已修复**：`OnlineWorker.score`/`score_batch` miss 返回全零 logits + `kv.miss` 打点，单/批一致、不抛异常（`pipeline.py`） | 「陈旧读」降级路径待 M6 与 G4 一并考虑 |
| **P1** | 无服务发现 / 模型版本注册中心：PS/datasystem 的 host/port 硬编码，无版本→checkpoint/表版本映射与灰度开关 | `embedding_ps_client.py`（默认 127.0.0.1:8000）、`datasystem_adapter.py`（默认 127.0.0.1:31501） | 引入轻量服务发现 + 模型注册 + 配置/灰度开关 |

---

## 6. 提交策略

- **粒度**：按「修改点 / 功能」独立提交，commit message 用中文「类型: 描述」前缀（`fix:` / `feat:` / `perf:` / `docs:`）。
- **频率**：每个功能点/修复点完成即提交并推送到远端 `origin`（用户约定：代码与文档每次修改后直接 commit + push）。
- **文档**：`docs/` 变更随对应功能同批或独立提交。