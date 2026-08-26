# 序列 Transformer 精排系统 · 工程级详细设计

> 版本：v0.3（三次修订）
> 上游：[端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6）、[详细设计](./detailed_design.md)（KV/Tensor/指标契约）
> 本文承接前两份文档，聚焦**工程侧**的高并发、分布式落地方案：线程模型、独立参数服务器、两阶段 brpc 分离部署、KV 零拷贝数据面、一致性哈希路由/元数据失效、动态 batching、可靠性、可观测性。

---

## 0. 文档关系与定位

| 文档 | 关注点 |
|---|---|
| e2e_design_spec.md | 业务边界、数据 fabric、设计决策（D1~D6） |
| detailed_design.md | KV 存储接口契约、Tensor 契约、算法→工程映射、指标埋点 |
| **engineering_design.md（本文）** | **软件架构 + 并发 + 部署 + 数据面优化 + 可靠性 + 可观测性** |

本文不重复前文的接口/张量契约；新增「怎么在高并发、分布式环境下正确、可靠、可观测地跑起来并可持续演进」的方案，并标注**已实现与尚未实现**的边界。

---

## 1. C++ / Python 职责分离

生产环境的**热路径必须有确定性、低时延、可横向扩展**，因此按如下原则拆分：

| 职责 | Python（单机参照实现） | C++（生产，brpc + bthread） |
|---|---|---|
| 模型结构定义 | ✓（`onetrans/nn/`） | 移植为 vLLM 自定义 op / 原生层 |
| 权重加载（checkpoint + seed 兜底） | ✓（`serving/weight_loader.py`） | 同语义的 checkpoint 读取器 |
| Nearline（Stage I）/ Online（Stage II）热路径 | ✓（`serving/pipeline.py`、`two_stage.py`） | **生产**，brpc worker |
| 稀疏 Embedding 参数服务 | 本地基准（`serving/embedding_ps_client.py`） | **生产**，独立 PS 服务（`deploy/ps/`） |
| UserKV 存储 | 本地/mmap 参照（`serving/local_adapter.py`） | datasystem 客户端封装（`serving/datasystem_adapter.py`） |
| 数值基准 / 等价性校验 | ✓（`serving/demo.py`） | 同一基准回放，保证移植数值一致 |

**核心结论**：Python 侧只承担「模型结构 / 权重加载 / 数值基准」三类职责；在线/近线热路径与稀疏参数服务在生产上由 C++（brpc + bthread M:N）承载。Python 实现是 C++ 移植的**黄金数值基准**（`demo.py` 等价性断言收敛到 1e-6 量级）。

---

## 2. 两阶段 brpc 分离部署

### 2.1 部署拓扑

```
                        行为流(按 user 分区)               请求(按 user 路由)
                             │                                    │
  ┌──────── Nearline Pool（C++ brpc 服务）────────┐   ┌──────── Online Pool（C++ brpc 服务）────────┐
  │  Stage I：tokenize + S 侧逐层编码 → 写 UserKV │   │  Stage II：读 UserKV + NS 编码 + 交叉打分    │
  └───────────────┬───────────────────────────────┘   └───────────────┬──────────────────────────────┘
                  │ put/append                            │ get/mget (+ 元数据校验, checksum/版本)
                  ▼                                       ▼
   ┌───────────────────────────  UserKV datasystem（fabric ②，按 user 分片）──────────────────────────┐
   └──────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                          │ 稀疏特征查表（S/NS 共享）
                                          ▼
   ┌─────────────── 独立稀疏参数服务器 PS（C++ brpc，fabric ①）──────────────┐
   │  多表分片嵌入表（细粒度锁）+ 版本号 ── 供 Nearline / Online 共享查表       │
   └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么要拆开部署

- **负载形态不同**：近线「写密集、批量尾巴」，在线「读密集、低时延、高 QPS」，拆开可独立扩容/限流/灰度。
- **稀疏参数独立服务化**：稀疏表可超单机内存，且每精排实例各自维护一份会成倍浪费内存/存储。独立 PS 让 Nearline/Online 共享同一份表。
- **UserKV 与 owner 同节点共存**：一致性哈希把 user 路由到 owner，读命中本地缓存，最大化数据本地性。

---

## 3. 计算面线程模型

### 3.1 单机参照实现（Python）

代码位置 `onetrans/serving/dispatcher.py`，与 brpc+bthread 一一对应：

| Python 参照 | 生产 brpc+bthread | 说明 |
|---|---|---|
| `Dispatcher.submit` | RPC 入口（bthread handler） | 分配 `req_seq`，按 user 稳定哈希/轮询选 worker |
| `WorkerPool`（N 个独立有界队列） | N 个 bthread worker（M:N） | 每 worker 独立队列，**无全局锁** |
| `Future` + `_inflight[req_seq]` | brpc `Controller` 回调 + 序列号 | **req_seq 异步匹配**，支持乱序完成 |
| `OverloadRejected`（队列满即拒） | brpc 背压（限流/拒绝/降级） | 有界队列满即快速失败，避免雪崩 |

关键设计选择：消全局锁（每 worker 独立有界队列 + 稳定哈希派发）；`req_seq` 异步匹配（乱序完成/超时取消）；背压（有界队列满即拒，上层重试/降级）。

### 3.2 与 LLM 线程模型的差异

第二阶段**非自回归**（M 候选整批并行），无 decode 逐 token 循环，故不需要 vLLM 的 PD 分离调度；重心是「高并发读 KV + 交叉注意力吞吐」。

> **待补强**：当前 `Dispatcher`/`WorkerPool` 无客户端超时、无健康度/熔断统计、无优雅排空（drain）语义，仅 `stop()` join。生产对应物需与 brpc 的 `bthread_id_lock`/`ExecutionQueue` 结合补齐（见 §8 可靠性）。

---

## 4. 独立稀疏参数服务器（PS）

### 4.1 设计要点

- **按 id 稳定哈希分片**（N 分片，每分片一把锁）：读写细粒度并发，无全局锁。
- **表版本 `version` 随写递增**：供「权重版本化加载 / 失效校验」。
- **未命中回 0 向量，客户端 seed 兜底哈希嵌入重建**：权重版本化最差路径。
- **多表（多模型版本）**：逻辑上以 `table` 区分（Python 侧 `LocalEmbeddingPS` 已支持）。

### 4.2 组件与 wire 契约

| 文件 | 职责 |
|---|---|
| `deploy/ps/embedding_service.proto` | RPC 契约：`Lookup` / `BatchLookup`，字段含 table/ids/dim/weights/version/shard_id |
| `deploy/ps/embedding_server.cc` | brpc+bthread 参考实现：`ShardedEmbeddingTable` + `EmbeddingServiceImpl` |
| `deploy/ps/BUILD` | bazel 构建（brpc/protobuf/glog/gflags） |
| `onetrans/serving/embedding_ps_client.py` | Python 侧：`ShardedEmbeddingTable` + `LocalEmbeddingPS` + `EmbeddingPSClient` |

### 4.3 已知契约缺陷（需修正）

1. **跨语言分片哈希不一致（P1）**：Python `shard_of` 用 `hash64(str(id))`（sha256），C++ `ShardOf` 用 Knuth 乘法哈希 `(id * kKnuth) % n`。二者**并不等价**（`embedding_server.cc` 注释误称「同构」）。生产若由 Python 客户端自行分片路由到具体 server shard，将与本设计（server 内部分片）不一致。**建议**：以 C++ Knuth 哈希为唯一标准，Python 侧 `ShardedEmbeddingTable.shard_of` 改为同款乘法哈希，并对 `str(id)`/负 id 语义对齐。
2. **C++ 侧单表（P1）**：`embedding_server.cc` 仅实例化一张 `ShardedEmbeddingTable`，`DoLookup` 忽略 `req.table()`（只回显），无法多模型版本/多表；与 Python 多表能力不一致。**建议**：server 侧用 `table -> ShardedEmbeddingTable` 映射（表注册/淘汰 + 版本），否则无法灰度新旧版本。

### 4.4 客户端数据面

`EmbeddingPSClient`：`local` 进程内分片表（基准）；`remote` 发 brpc RPC。当前 `remote` 在 Python 侧为**占位**（`NotImplementedError`），生产 brpc 调用发生在 C++ worker 内。

---

## 5. KV 零拷贝 UserKV 数据面

### 5.1 拷贝点识别

1. **序列化读侧**：`bytes` → `Tensor` 常规做法逐层 `copy()`，产生一次全量 CPU 拷贝。
2. **进程间传输**：nearline 写 / online 读之间经 socket/UDS 逐字节搬运 + 再分配。

### 5.2 消除方案

| 环节 | 手段 | 落地点 |
|---|---|---|
| 序列化读侧 | 预分配 buffer + `torch.frombuffer` 零拷贝视图 + 逐层 offset | `serialize.py` |
| 序列化写侧 | 预分配 `bytearray` + `ctypes.memmove` 直搬底层内存 | `serialize.py` |
| 进程间传输 | `mmap`（`ACCESS_WRITE` 双向零拷贝） | `local_adapter.py` |

`demo.py:test_zero_copy` 作回归。生产由 datasystem HBM/DRAM + 卡间直通承接。

### 5.3 数据面元数据一致性缺口（P1，正确性）

`serialize.py` 的 header 只存 **dtype + shape**，**不存有效长度**（`s_len` / `per_layer_len`）。而 `LocalKVStore` 依赖「record 对象字段」保留这些值，`YuanrongKVStore.put` 只把 `rec.payload` 写进 datasystem（metadata 字段丢失）；`YuanrongKVStore.get` 用「全宽 shape」重建 `s_len`/`per_layer_len`——对**左 padding（用户历史短于 max_seq_len）**的用户，会误把 padding 当有效，导致在线交叉注意力掩码错误。

**建议**：把 `s_len` 与 `per_layer_len` 纳入 `serialize` 的 header（读写双方一致），或让 datasystem 后端 `get` 从元数据面（`KVPointer`）取回有效长度，并与 payload checksum 交叉校验。本地后端当前因对象内联元数据而正确，属「隐性依赖」，需固化到契约避免后端迁移引入回归。

---

## 6. 一致性哈希路由 + 元数据/版本失效

### 6.1 一致性哈希路由（数据本地性）

| 文件 | 职责 |
|---|---|
| `serving/router.py` | `JumpConsistentHash`（最小 remap）+ `RingHash`（虚拟节点，动态增删）+ `Router` 门面 |
| `serving/sharded.py` | `ShardedKVStore`：按 `user_id` 路由，`mget`/`delete`/`prefetch` 按 shard 聚合 |

> **路由哈希不统一（P1，一致性，三次审阅新增）**：数据本地性的前提是「KV owner shard」与「处理 worker」对同一 `user_id` 落到同一物理节点。当前三处路由用**三种不同算法，彼此不等价**：
> - `ShardedKVStore` 用 jump 哈希（`Router(num_shards)` → `JumpConsistentHash.shard_of`）；
> - `WorkerPool.worker_for` 用 `hash64(key) % num_workers`（取模，`dispatcher.py` L106-107）；
> - `ShardedEmbeddingTable.shard_of` 用 `hash64(str(feat_id)) % num_shards`（取模，且与 C++ Knuth 哈希不等价，见 §4.3）。
>
> 后果：即使 `num_shards == num_workers`，同一 user 的 KV 分片与 worker 也可能不一致，破坏「KV 与 worker 同节点共存」；且取模法在扩缩容时**全量 remap**（仅 jump 哈希最小化 remap）。**建议**：worker 分派与 KV 分片统一复用同一 `Router`（同一 `hash64` + 跳变哈希），PS 分片统一到 C++ Knuth 哈希。

### 6.2 元数据/版本失效

| 文件 | 职责 |
|---|---|
| `serving/meta_store.py` | `KVPointer`（version/len/checksum/ts/obj_key）+ `LocalMetaStore`（TTL 惰性过期） |
| 一致性校验 | Online 读 KV 校验 `checksum == pointer.checksum`（`validate_pointer`） |

### 6.3 乐观并发 append

增量 append 携带 `offset` 校验；冲突即拒，由上层重放/全量 put 兜底。

> **待补强**：`datasystem_adapter.append` 是「读-合并-写」三段式，`offset` 校验为进程内乐观并发，**非跨进程原子**（存在 TOCTOU 窗口）。生产需依赖 datasystem 的原子 CAS/版本写，或 nearline 侧按 user 的分布式排他（如 fencing token）。（详见 §8）

---

## 7. 动态 batching

| 文件 | 职责 |
|---|---|
| `serving/pipeline.py` | `BatchScheduler`（FIFO 攒批）+ `score_batch`（批量 mget + 打包）、`OnlineWorker` |
| `serving/two_stage.py` | `score_ns_batch`（逐层 stack，左 padding 掩码，与 `score_ns` 数值等价） |

- **攒批窗口**：`max_wait_seconds` 内未满批按已攒 ≥1 条返回，时延有界。
- **数值等价**：逐层宽度 `S_l=dims[l]` 恒定，逐层 stack（左 padding）与逐条打分在 B=1 时逐位一致。

---

## 8. 可靠性设计（现状 + 缺口）

| 能力 | 现状 | 缺口 / 建议 |
|---|---|---|
| 客户端超时 | 无（`Future` 不设 deadline） | KV/PS/RPC 调用加超时，过时 cancel |
| 重试 & 幂等 | 无 | 读操作幂等可安全重试；写需幂等键/版本 |
| 熔断/限流 | 仅队列背压（`OverloadRejected`） | 按后端错误率熔断 + 令牌桶限流 |
| 健康检查/探针 | 无 | 提供 `/healthz` + 依赖探针（datasystem/PS） |
| 优雅停机/排空 | `WorkerPool.stop()` join，无 drain | 先停收新请求再排空队列（drain & wait） |
| 一致性（append） | `offset` 乐观校验 | datasystem 原子 CAS / fencing token（见 §6.3） |
| 降级路径 | seed 兜底、miss 兜底（已有） | 明确「陈旧读 + 打点」与「空结果快速返回」策略 |

> 上表为工程级落地必须补齐的横切能力，多数在当前单机参照中尚未实现，属生产验收项。

---

## 9. 可观测性设计（现状 + 缺口）

| 能力 | 现状 | 缺口 / 建议 |
|---|---|---|
| 指标 | `ServingMetrics`（内存直方图/计数/仪表，p50/p99） | 接 Prometheus/OTel：分桶直方图（当前全样本存储）、导出 endpoint |
| 日志 | 无结构化日志 | 结构化日志（req_id/trace_id/user_id/耗时），分级 |
| 追踪 | 无 | 分布式 trace（Span：路由/读 KV/打分/PS 查表）贯穿 Nearline→Online |
| 事件 | `metrics.count` 打点已有 | 补齐 KV miss/hit、背压拒绝、降级等关键事件告警 |

> `detailed_design.md` §6 已定义埋点命名约定；本文重点是「埋点出口」尚未实现，仅内存收集。

---

## 10. 组件 / 文件清单

```
onetrans/serving/
├── serialize.py            KV 序列化（读侧 frombuffer 零拷贝）
├── local_adapter.py        本地 KV（mmap 共享内存）
├── kv_store.py             KV 统一接口 + 后端工厂
├── meta_store.py           元数据/版本/TTL
├── router.py               一致性哈希（jump/ring）
├── sharded.py              分片 KV 门面
├── dispatcher.py           计算面线程模型（Dispatcher/WorkerPool/背压）
├── embedding_ps_client.py  稀疏 PS 客户端（本地/remote）
├── datasystem_adapter.py   datasystem 客户端封装（fabric ②）
├── pipeline.py             Nearline/Online 管道 + BatchScheduler
├── two_stage.py            两阶段推理（Stage II 交叉打分 + 攒批）
├── weight_loader.py        checkpoint 权重加载 + seed 兜底
└── demo.py                 端到端数值基准/等价性校验

deploy/ps/
├── embedding_service.proto  PS RPC 契约
├── embedding_server.cc      PS brpc+bthread 参考实现（单表）
└── BUILD                    bazel 构建
```

---

## 11. 生产落地缺口总表（P0/P1/P2 分级）

| 级别 | 缺口 | 位置 | 影响 |
|---|---|---|---|
| ~~**P1（正确性）**~~ | ✅ **M5 已修复**：`serialize` header 固化 `s_len`/`per_layer_len`，datasystem 读写语义与 local 一致 | `serialize.py` / `datasystem_adapter.py` | （已消除）子满长用户的在线打分错误 |
| ~~**P1（一致性）**~~ | ✅ 已修复：PS 跨语言分片哈希统一到 Knuth | `embedding_ps_client.py` vs `embedding_server.cc` | （已消除）分片路由不一致风险 |
| ~~**P1**~~ | ✅ 已修复：C++ PS 多表（`table→ShardedEmbeddingTable` + 版本） | `embedding_server.cc` | （已消除）无法多模型版本灰度 |
| **P1（可靠性）** | 无超时/重试/熔断/健康检查/优雅停机 | `dispatcher.py`/客户端 | 生产稳定性（M6） |
| ~~**P1（一致性）**~~ | ✅ **M5 已缓解**：`DeltaKV.expect_checksum` CAS fencing，`cas_conflict` 拒绝（datasystem 原生原子 CAS 仍可后置） | `datasystem_adapter.py` | （已缓解）并发写脏数据风险 |
| ~~**P1（一致性）**~~ | ✅ **M5 已修复**：`WorkerPool.worker_for` 复用 `Router`（jump 哈希），路由统一 | `sharded.py`/`dispatcher.py`/`embedding_ps_client.py` | （已消除）user 的 KV 与 worker 不共址、扩缩容全量 remap |
| **P1（可观测）** | 指标仅内存，无导出/日志/追踪 | `metrics.py` | 无法线上观测（M6） |
| **P1（落地）** | 无 C++ Nearline/Online 热路径 worker（仅 PS 有 C++ 参考实现），混合参数化层未移植 vLLM 自定义 op | `deploy/ps/`（仅 PS）、`nn/mixed_*` | 两阶段 brpc 分离部署仍为设计态（M8） |
| **P1（落地）** | tokenizer + 稀疏 embedding 查表未接入 serving 热路径（`ingest`/`score` 直接收已 tokenize 的 `s_emb`/`ns_emb`） | `pipeline.py` | 行为流→查表→编码、特征服务→查表→打分 未端到端接线（M7） |
| **P1（落地）** | ~~KV miss 硬失败无降级~~（✅ M5 已修复：miss 返回全零 + 打点）；无服务发现/模型版本注册中心（host/port 硬编码，M7） | `pipeline.py`、`embedding_ps_client.py`/`datasystem_adapter.py` | 不可灰度、容错部分已具备 |
| P2 | redis 后端、HBM 直通 | `kv_store.py`/`datasystem_adapter.py` | 环境依赖 |
| P2 | 无测试框架/CI | 全局 | 回归保障弱（`demo.py` 单脚本 assert） |
| P2 | `_project_ns`/`_apply_ns_ffn` 逐 token 循环、`RingHash` 建环 O(n²)、percentile 全样本 | `two_stage.py`/`router.py`/`metrics.py` | 局部性能（Ns 小，可容忍） |

> 上述分级对应「实现 & 现状」文档的差距评估章节；M5（G1/G2/G3/G8）已完成正确性收口，剩余缺口按 M6（可靠+观测）→ M7（热路径接线）→ M8（C++ 移植）推进，详见 `gap_analysis.md` 第四部分路线图。