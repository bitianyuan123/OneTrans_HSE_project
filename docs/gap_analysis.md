# 剩余差距审视与高优先级项详细设计

> 版本：v0.1
> 上游：
> - [端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6、三 fabric）
> - [详细设计](./detailed_design.md)（KV/Tensor 接口契约、逐组件设计、指标采集点、§8 里程碑 M0~M4）
> - [工程级详细设计](./engineering_design.md)（软件架构/线程模型/部署/可靠性/可观测性，§11 缺口总表）
> 关联：[实现 & 现状总结](./implementation_status.md)（§5 差距评估：正确性/可靠性/可观测性/工程化/集成落地）

本文对**已识别的剩余差距**做一次收敛性审视后，对**高优先级、尚未实现**的项做必要性分析与可落地的详细设计，并给出分阶段路线图。本文**不写代码**，所有签名/行号以仓库当前代码为准（基线分支 `feat/onetrans-e2e-serving`）。

---

## 第一部分：差距总览与再定位

### 1.1 总表

下表重新归类 `implementation_status.md` §5 与 `engineering_design.md` §11 的全部已识别差距，并给出重新评估后的优先级与「是否已被并行工作项处理」。优先级含义：

- **P1-High**：正确性/可靠性/端到端闭环的硬前提，**必须现在做**（阻断切换 datasystem、阻断线上可运行）。
- **P1-Mid**：不影响数值正确性，但有明确的负载/灰度/排障缺口，可按依赖顺序**紧随其后的里程碑**做。
- **P2**：非阻断，可后置的工程化/性能项。

| # | 级别 | 类别 | 差距（一句话现状） | 来源 | 重新评估优先级 | 并行工作项已处理 |
|---|---|---|---|---|---|---|
| G1 | P1 | 正确性 | datasystem 后端写/读只经过 `payload`，丢失 `s_len`/`per_layer_len`，左 padding 用户在线掩码错误（`LocalKVStore` 靠 record 内联元数据才正确，属隐性依赖） | imp §5.1、eng §11 | （保留必要性分析，M5 已实现） | **是**（M5） |
| G2 | P1 | 一致性 | datasystem `append` 是「读-合并-写」，`offset` 乐观校验为进程内逻辑，跨进程存在 TOCTOU 窗口 | imp §5.2、eng §6.3/§11 | （保留必要性分析，M5 已实现） | **是**（M5） |
| G3 | P1 | 一致性/本地性 | 路由哈希三处不统一（KV 分片 jump vs worker 取模 `hash64%n` vs PS Knuth），破坏「KV 与 owner worker 同节点共存」，扩缩容全量 remap | imp §5.1、eng §6.1/§11 | （保留必要性分析，M5 已实现） | **是**（M5） |
| G4 | P1 | 可靠性 | 无客户端超时、无重试&幂等、无熔断/限流、无健康检查/优雅停机（仅队列背压） | imp §5.2、eng §8/§11 | **P1-High** | 否 |
| G5 | P1 | 可观测性 | 指标仅进程内存（`ServingMetrics`），无导出/无结构化日志（缺 req_id/trace_id）/无分布式 trace | imp §5.3、eng §9/§11 | P1-Mid | 否 |
| G6 | P1 | 落地 | 无 C++ Nearline/Online 热路径 worker（仅 PS 有 C++ 参考），混合参数化层未移植 vLLM 自定义 op | imp §5.5、eng §11 | P1-Mid | 否 |
| G7 | P1 | 落地 | tokenizer + 稀疏 embedding 查表未接入 serving 热路径（`ingest`/`score` 直收已 tokenize 的 `s_emb`/`ns_emb`） | imp §5.5、eng §11 | **P1-High** | 否 |
| G8 | P1 | 可靠性/降级 | KV miss 硬失败（`raise KeyError`），无「陈旧读+打点」「空 KV 快速返回」降级 | imp §5.5、eng §8 | （保留必要性分析，M5 已实现） | **是**（M5） |
| G9 | P1 | 落地 | 无服务发现 / 模型版本注册中心，PS/datasystem host/port 硬编码，无版本→checkpoint/表版本映射与灰度开关 | imp §5.5、eng §11 | P1-Mid | 否 |
| G10 | P1 | 一致性 | PS 跨语言分片哈希不等价（Python `hash64(str(id))` vs C++ Knuth） | imp §5.1、eng §4.3/§11 | （保留必要性分析，设计已实现） | **是** |
| G11 | P1 | 落地 | C++ PS 仅单表，`DoLookup` 忽略 `req.table()`，无法多模型版本/灰度 | imp §5.1、eng §4.3/§11 | （保留必要性分析，设计已实现） | **是** |
| G12 | P2 | 落地 | redis 后端、datasystem HBM 直通（异构对象）未实现 | eng §11 | P2 | 否 |
| G13 | P2 | 工程化 | 无测试框架/CI（`demo.py` 脚本 assert 即回归） | eng §11、imp §5.4 | P2 | 否 |
| G14 | P2 | 性能 | `_project_ns`/`_apply_ns_ffn` 逐 token 循环、`RingHash` 建环 O(v·n²)、percentile 全样本 | eng §11、imp §5.4 | P2 | 否 |

> 来源缩写：imp = `implementation_status.md`，eng = `engineering_design.md`。

### 1.2 与本会话并行工作项的边界

**P1 差距中已由本会话实现**（代码已落地并经 `demo.py` 端到端校验，本文第二部分保留其必要性分析、第三部分保留其详细设计作为设计记录）：

- **G1（元数据固化，M5）**：`serialize` header 显式纳入 `s_len`/`per_layer_len`（`serialize.py` 新增 `read_header`/`deserialize_with_meta`，向后兼容旧 payload）；`NearlineWorker.ingest` 写入时带上有效长度；`YuanrongKVStore.get`/`append` 经 `deserialize_with_meta` 读回，datasystem 后端不再依赖全宽 shape 重建。
- **G2（append CAS fencing，M5）**：`DeltaKV` 新增 `expect_checksum`（fencing token），local/datasystem 两侧 `append` 在 offset 校验后追加 checksum CAS 校验，不匹配以 `cas_conflict` 拒绝，消除读-合并-写 TOCTOU 丢写窗口。
- **G3（路由统一，M5）**：`WorkerPool.worker_for` 从 `hash64 % num_workers` 改为复用 `Router`（jump 一致性哈希），worker 分派与 KV 分片对同一 user 落同一桶，保证数据本地性与最小 remap。
- **G8（KV miss 降级，M5）**：`OnlineWorker.score`/`score_batch` miss 时返回全零 logits（`kv.miss` 打点），命中行正常打分，保持 `[B, T]` 展平顺序，不再 `raise KeyError`。

**另两条 P1 已由本会话并行的 C++ 数据面客户端工作项实现**：

- **G10（PS 跨语言分片哈希不等价）**：哈希已统一到 **Knuth 乘法哈希**——`/workspace/deploy/ps/embedding_server.cc` 的 `detail::ShardOf` 为唯一标准，Python 侧 `/workspace/onetrans/serving/embedding_ps_client.py` 的 `ShardedEmbeddingTable.shard_of` 改为同款 Knuth 乘法哈希（复用 `hash64` 不再 `str(id)` 混淆、负 id 语义对齐）。详细设计见代码注释，本文不重复设计。
- **G11（C++ PS 仅单表）**：`embedding_server.cc` 已从单 `ShardedEmbeddingTable` 改为 `table -> ShardedEmbeddingTable` 多表映射（表注册/淘汰 + 版本），`DoLookup` 按 `req.table()` 路由。详细设计见代码注释，本文不重复设计。

本文**第二部分**对 G1/G2/G3/G8/G10/G11 仍保留必要性分析（标注「已处理」），**第三部分**不再为其展开详细设计；**重点对象是 G4~G7、G9 尚未实现的项**。

---

## 第二部分：高优先级项必要性分析

> 每条按：问题本质 / 若不做的在线后果 / 触发条件 / 影响面 / 为什么现在必须做或可后置。均为可证伪的具体后果。

### 2.1 G1：datasystem 后端丢失 `s_len` / `per_layer_len`（正确性，最高优先级）

- **问题本质**：`LocalKVStore`（`/workspace/onetrans/serving/local_adapter.py`）以 `UserKVRecord` 对象为存储单位，`put` 直接把 `rec`（含 `s_len`/`per_layer_len` 字段）存进内存、`get` 原样返回，因此 keep 有效长度。而 `YuanrongKVStore.put`（`/workspace/onetrans/serving/datasystem_adapter.py` L56-59）只执行 `kv().set(str(rec.key), rec.payload)`，`s_len`/`per_layer_len`/`checksum` 全部丢弃；`get`（L61-77）则用 `per_layer[0][0].shape[1]` 重建 `s_len`、用 `[k.shape[1] for ...]` 重建 `per_layer_len`。由于 `serialize` 的 header 只存 `dtype`+`shape`（`/workspace/onetrans/serving/serialize.py` L56-63），`K_s^l` 的 `shape[1]` 是 pyramid 该层**满宽 `dims[l]`**，无法区分「有效 token 数」与「左 padding」，于是左 padding 用户的 `per_layer_len` 被误读为满宽。
- **若不做的在线后果**：`decode_record`（`/workspace/onetrans/serving/pipeline.py` L196-201）把 `rec.per_layer_len` 交给 `score_ns`/`score_ns_batch` 构造 `s_mask`（`/workspace/onetrans/serving/two_stage.py` L140-147 与 L196-202）。一旦 `per_layer_len[l]` 被读成满宽，padding 位置被当作真实 token 进入交叉注意力，K/V 里的 padding 噪声被计入 softmax 与后续 FFN，**历史短于 `max_seq_len` 的用户得分系统性错误**；且错误是「静默」的——只有与单前向对照（`demo.py` 的 `test_equivalence`）才会暴露，线上无任何信号。
- **触发条件**：凡「用户历史有效长度 < `max_seq_len`」即触发（左 padding 语义下这是常态而非边缘情况）。当前 `demo.py` 用 `local` 后端跑通，故未触发；一旦切 `backend="datasystem"` 即 100% 触发。
- **影响面**：所有走 datasystem 后端的在线打分链路；先单卡跑通再切 datasystem 的迁移路径。
- **为什么现在必须做**：这是「先单卡跑通、再切 datasystem」路径上**最大的隐藏正确性风险**——`LocalKVStore` 靠 record 对象内联元数据才正确，是**隐性依赖**，未固化到序列化契约。若不做，切换后端的当天就会出现「短历史用户分数错误」且难以归因。必须将其显式化到 `serialize` header 或读侧元数据校验，并把验收写进 M5。

### 2.2 G2：datasystem `append` 非原子（一致性）

- **问题本质**：`YuanrongKVStore.append`（`/workspace/onetrans/serving/datasystem_adapter.py` L82-108）是「读-合并-写」三段式：`get` → 校验 `delta.offset == rec.s_len` → `kv().set`。`offset` 校验是**进程内**乐观并发；两个 nearline 进程（或同进程两次并发 append）之间无原子栅栏，存在 **TOCTOU**：A 进程 `get` 得 `s_len=10`，B 进程同时 `append` 到 13，A 再 `set` 覆盖为 `10+ΔL`，**丢失 B 的 3 条**且无异常。
- **若不做的在线后果**：某用户行为流多次触发同一 `append` 的并发场景下，KV 被静默截断/回退，`pointer.checksum` 与 payload 漂移，online 读侧校验失败或读到陈旧 K/V，导致该用户分数错误；且「丢写」无告警。
- **触发条件**：同一 user 的 append 由多进程/多副本并发执行（nearline 单 owner 串行写时不触发；但 failover、任务重分配、重复投递都会打破串行假设）。
- **影响面**：Nearline 写路径（fabric ②），用户级（单 user 数据）。
- **为什么现在必须做 / 可后置**：在「nearline 按 user 单 owner 串行写 + 冲突即全量重建」的当前部署下可短期容忍，故排 **P1-Mid**；但它是可靠性四件套（G4）中「写幂等/fencing」的正确性根基，应在 M5 与 G1 一并固化，避免把隐患留给后续加副本。

### 2.3 G3：路由哈希不统一（破坏 KV/worker 同节点数据本地性）

- **问题本质**：三处路由算法互不等价——KV 分片用 jump 哈希（`/workspace/onetrans/serving/sharded.py` L35、`/workspace/onetrans/serving/router.py` 的 `Router`→`JumpConsistentHash.shard_of`，基于 `hash64` sha256）；worker 分派用 `WorkerPool.worker_for = hash64(key) % num_workers`（`/workspace/onetrans/serving/dispatcher.py` L106-107，取模）；PS 分片用 Knuth（并行项已统一）。`Dispatcher._choose_worker`（`dispatcher.py` L205-211`）在 `mode="hash"` 下直接调 `worker_for`。
- **若不做的在线后果**：即使 `num_shards == num_workers`，同一 `user_id` 经 jump 哈希与取模得到的桶号**大概率不相同**，导致该 user 的 KV 落在节点 X、其处理 worker 落在节点 Y，每次在线打分都跨节点读 KV。后果：① 数据本地性被破坏，KV 读时延从「本地命中」退化到「跨节点 RDMA/RoCE」，`online.kv_get` p99 显著抬升；② 扩缩容时取模法**全量 remap**（近 100% 键迁移），而 jump 哈希 remap 仅 ~1/k（`demo.py` 实测 8→9 桶 remap≈0.116）。
- **触发条件**：凡 `mode="hash"` 且 `num_workers>1` 即触发；`num_shards≠num_workers` 时进一步叠加分片粒度不一致。
- **影响面**：所有高 QPS 在线打分链路，直接决定「KV 与 worker 同节点共存」这个核心架构收益（`engineering_design.md` §2.2）是否成立。
- **为什么现在必须做**：它是 G7 端到端接线与后续 scale-out 的**路由统一底座**；改动小（收敛到同一 `Router`）、收益大（本地性 + 最小 remap），且是避免「上线即跨节点风暴」的硬前提，故 **P1-High**。

### 2.4 G4：可靠性四件套缺失（超时/重试&幂等/熔断限流/健康检查优雅停机）

- **问题本质**：`Future` 无 deadline（`/workspace/onetrans/serving/dispatcher.py` 的 `submit` 只做背压，不设超时）；无重试与幂等约定；仅队列背压（`OverloadRejected`）无后端错误率熔断与令牌桶限流；`WorkerPool.stop()` 仅 join 无 drain（`dispatcher.py` L94-103）。
- **若不做的在线后果**：① 某个后端（datasystem/PS）变慢或假死时，调用方无限等待，bthread 被耗尽，雪崩式拖垮整个池；② 瞬时错误无法重试，读侧一次抖动即失败；③ 依赖故障无隔离，一个坏后端拉垮全链路；④ 发版/缩容时队列里的请求被直接丢弃，造成可避免的失败与指标突刺。
- **触发条件**：后端抖动/慢节点/网络分区/版本发布/实例摘除——生产常态。
- **影响面**：全体在线/近线链路，`dispatcher.py`/`datasystem_adapter.py`/`embedding_ps_client.py`。
- **为什么现在必须做 / 可后置**：不影响数值正确性，但在「工程级可用」是必补项（eng §8 已列为生产验收项）。正确性（G1/G3）优先，故排在其后的 **P1-High** 里程碑（M6）：先补超时与健康检查（最小闭环），再补重试/熔断/优雅停机。

### 2.5 G5：可观测性缺口（指标仅内存、无结构化日志、无 trace）

- **问题本质**：`ServingMetrics`（`/workspace/onetrans/serving/metrics.py`）`_Histogram` 存全样本、`snapshot()` 只在本进程打印，无 Prometheus/OTel 导出；无结构化日志（无 req_id/trace_id/user_id 关联）；无分布式 trace 贯穿 Nearline→Online→PS/datasystem。
- **若不做的在线后果**：线上出现「某用户分数异常/某 shard 慢/某批丢失」时**无法归因**——看不到单请求的 KV 命中、PS 查表、背压拒绝、降级路径，无法定位到具体 user/req；百分位全样本排序随样本量线性膨胀，长跑内存增长。
- **触发条件**：任何线上异常/容量评估/灰度验证都需要。
- **影响面**：排障与容量评估能力，不直接影响运行时正确性。
- **为什么现在必须做 / 可后置**：可后置于正确性，但必须在「上线负载实验（M3）」之前落地，否则实验数据不可信。**P1-Mid**，随 M6（观测）一并做，最小编码是「分桶直方图 + req_id 贯穿 + 指标导出 endpoint」。

### 2.6 G6：无 C++ Nearline/Online 热路径 worker

- **问题本质**：两阶段 brpc 分离部署目前仅 **PS**（`/workspace/deploy/ps/`）有 C++ 参考实现；Stage I/II 热路径与混合参数化层（`/workspace/onetrans/nn/attention/mixed_attention.py` 的 `MixedCausalSelfAttention`、`/workspace/onetrans/nn/ffn/mixed_ffn.py` 的 `MixedFFN` 的逐 token `W_ns_list`/`networks_ns_list`）仍只在 Python（`/workspace/onetrans/serving/two_stage.py`）运行，未移植 vLLM 自定义 op。
- **若不做的在线后果**：生产高 QPS 下的时延/吞吐由 Python 解释器与逐 token 循环决定，无法达到 brpc+bthread 的确定性低时延与横向扩展；`engineering_design.md` §1 的「C++ 生产 / Python 基准」分工只完成了一半，两阶段分离部署停留在设计态。
- **触发条件**：负载压测/生产切流时暴露。
- **影响面**：全体热路径（`encode_s`/`score_ns`/`score_ns_batch`），是「单卡参照已通 → 集群落地」的最后形态。
- **为什么现在必须做 / 可后置**：工程量大、依赖 vLLM 自定义 op 与昇腾/GPU 环境，**可分阶段后置**；但每阶段必须以 Python 为黄金数值基准。排 **P1-Mid**，作为最后一个里程碑（M8）分 M8a→M8c 三层推进，M7 的端到端正确链路（Python）先闭环。

### 2.7 G7：tokenizer + 稀疏 embedding 查表未接入 serving 热路径

- **问题本质**：`NearlineWorker.ingest` 直接收已 tokenize 的 `s_emb`（`/workspace/onetrans/serving/pipeline.py` L44-55），`OnlineWorker.score` 直接收 `ns_emb`（L89-99）；而 `OneTransTokenizer.encode_s/encode_ns`（`/workspace/onetrans/nn/tokenizer.py` L135/L143）与 `EmbeddingPSClient.lookup`（`/workspace/onetrans/serving/embedding_ps_client.py` L143）都没有接到 worker 之前。行为流→查表→编码、特征服务→查表→打分 未端到端接线。
- **若不做的在线后果**：系统只能消费「人造已 tokenize 张量」，无法从真实行为事件流与候选特征服务驱动，**端到端链路不闭环**，无法做真实负载/容量实验，也无法与 DeepFM/DCNv2 在同 seam（同 embedding）公平对比。
- **触发条件**：接入真实事件流/特征服务即触发。
- **影响面**：「行为流→S 编码→写 KV」「候选特征→NS 编码→打分」两条入口链路（fabric ① 到 fabric ②）。
- **为什么现在必须做**：改动相对轻量（在 worker 外接一层**接口注入**的 tokenize+lookup 前置），且是 G6 C++ 移植与 M3/M4 实验的**前提**（没有它，负载实验的输入不真实）。故 **P1-High**，排 M7。

### 2.8 G8：KV miss 硬失败（`raise KeyError`）无降级

- **问题本质**：`OnlineWorker.score`（`/workspace/onetrans/serving/pipeline.py` L103-105）与 `score_batch`（L127-130）在 `store.get` 返回 `None` 时直接 `raise KeyError`，将单请求失败向上抛给调用方，无「陈旧读+打点」或「空 KV 兜底快速返回」。
- **若不做的在线后果**：冷启动用户、TTL 过期、分片迁移瞬间、KV 淘汰——这些**常态**事件都会变成请求失败/异常堆栈，错误率被无谓抬高，且没有区分「KV 真缺」与「读侧校验失败」的降级路径。
- **触发条件**：任何 KV miss（冷用户、过期、搬迁、极端 miss 突发）。
- **影响面**：在线打分链路，冷启动用户与灰度窗口。
- **为什么现在必须做 / 可后置**：正确性上 `raise` 本身没错，但缺降级会在灰度/冷启动期放大失败率。**P1-Mid**，随 M5 与 G1 一并做（降级语义依赖 G1 的元数据校验结果），提供 G8 的两条降级路径（见第三部分 3.8）。

### 2.9 G9：无服务发现 / 模型版本注册中心

- **问题本质**：PS 默认 `127.0.0.1:8000`（`/workspace/onetrans/serving/embedding_ps_client.py` L130-131）、datasystem 默认 `127.0.0.1:31501`（`/workspace/onetrans/serving/datasystem_adapter.py` L30）硬编码；无「model_version → checkpoint 路径 / PS 表版本 / 灰度开关」的注册与映射。
- **若不做的在线后果**：无法灰度发布新模型版本（切版本要改代码/重启）、无法按版本路由/回滚、实例部署位置变化需手动改配置，规模化运维成本与出错率陡增。
- **触发条件**：多实例部署、多模型版本并行、灰度发布即触发。
- **影响面**：部署/灰度/回滚链路（fabric ③ 控制面）。
- **为什么现在必须做 / 可后置**：单实例硬编码可先跑通，故 **P1-Mid**；但它是 G6/G7 之后的灰度前提，随 M7 引入轻量服务发现 + 模型注册（见第三部分 3.9）。

### 2.10（已处理，保留必要性分析）G10：PS 跨语言分片哈希不等价

- **必要性**：一致性哈希分片的**首要要求**是「读写两侧对同一 id 落同一分片」。若 Python 客户端用 sha256、C++ server 用 Knuth，则客户端自行分片路由（或预判分片）与 server 内部分片不一致，轻则跨分片错读、重则「应命中的 key 读不到 / 写错分片」。**统一到唯一标准（C++ Knuth）是与 PS 正确交互的硬前提**。——已由并行工作项实现，详见 `/workspace/deploy/ps/embedding_server.cc` 与 `/workspace/onetrans/serving/embedding_ps_client.py` 代码注释。

### 2.11（已处理，保留必要性分析）G11：C++ PS 仅单表

- **必要性**：生产需多模型版本/**灰度**并存。单表意味着同一时刻只能服务一个 `table`，`req.table()` 被忽略（回显而已），无法「新旧版本并存 + 按版本查表 + 权重版本化失效」。**表维度是版本化/灰度的硬前提**。——已由并行工作项实现（`table -> ShardedEmbeddingTable` 映射），详见代码注释。

---

## 第三部分：高优先级项详细设计（尚未实现项 G1~G9）

> 以下所有「落点」均为**绝对路径**下真实存在的文件/函数（本会话已阅读核实）；设计对齐既有签名。

### 3.1 G1：KV 有效长度元数据（`s_len`/`per_layer_len`）固化

#### 目标

让 `UserKVRecord` 的有效长度元数据在**跨后端**（local / datasystem / redis）迁移时**不丢失**，使左 padding 用户经 datasystem `put→get` 后 `decode_record` 得到的 `per_layer_len` 与写入时逐层一致，进而 `score_ns`/`score_ns_batch` 的 `s_mask` 语义正确。

#### 为什么 `per_layer_len` 必须在 shape 之外显式编码

`K_s^l` 的 `shape[1]` 是 pyramid 该层的**满宽 `dims[l]`**（恒定），而左 padding 语义下「有效 token 数」=`per_layer_len[l]` ≤ `dims[l]`。二者不相等，故**只靠 shape 无法区分左 padding**，必须显式存储有效长度（这正是 `engineering_design.md` §5.3 所指缺陷）。`s_len` 是原始历史有效长度，用于 `append` 的 offset 语义；每层 `per_layer_len[l]` 用于该层交叉注意力掩码。

#### 候选方案

**方案 A（推荐，主）——`serialize` header 纳入有效长度，读写双方一致**

`/workspace/onetrans/serving/serialize.py` 的 `serialize` 当前仅写 `dtype/n_layers/layers[].k_shape/v_shape`。最小变化（header JSON）：

```json
{
  "dtype": "float16",
  "n_layers": 6,
  "s_len": 23,
  "layers": [
    {"l": 0, "k_shape": [1,100,4,64], "v_shape": [1,100,4,64], "len": 23},
    {"l": 5, "k_shape": [1, 10,4,64], "v_shape": [1, 10,4,64], "len": 10}
  ]
}
```

- `serialize` 签名扩展为 `serialize(per_layer, s_len: int|None=None, per_layer_len: list[int]|None=None) -> bytes`。
  - 当 `s_len`/`per_layer_len` 未传（旧调用点/无 padding 场景）时，**回退**为 `len = k_shape[1]`、`s_len = layers[0]["len"]`，保持旧行为与 `demo.py:test_serialize_roundtrip` 兼容。
- `deserialize` 不变（继续返回 `list[(K,V)]`，零拷贝视图）；新增只解析 header 的读函数，避免重复反序列化：

```python
def read_header(payload) -> dict:
    """仅解析 <magic><header_len><header_json>，返回 header dict（含 s_len/layers[].len）。"""
    # 复用现有魔数校验 + struct.unpack_from(_HEADER_FMT, ...) + json.loads

def deserialize_with_meta(payload) -> tuple[list[tuple[Tensor,Tensor]], int, list[int]]:
    hdr = read_header(payload)
    per_layer = deserialize(payload)          # 零拷贝视图
    s_len = hdr.get("s_len", per_layer[0][0].shape[1] if per_layer else 0)
    per_layer_len = [m.get("len", m["k_shape"][1]) for m in hdr["layers"]]
    return per_layer, s_len, per_layer_len
```

- `per_layer_offsets`（`serialize.py` L126）同步读取 `len` 时无需改动（只关心偏移）；但为一致性，其解析可与 `read_header` 共享。

**方案 B（补充，交叉校验）——datasystem `get` 从元数据面 `KVPointer` 取回有效长度 + checksum 交叉校验**

`YuanrongKVStore` 增加一个可选注入的 `MetaStore`（`/workspace/onetrans/serving/meta_store.py` 的 `KVPointer`/`validate_pointer`）：

- `get` 后：`ptr = self._meta.get(mv, uid)`；用 `ptr.s_len`/`ptr.per_layer_len` 覆盖重建值，并 `validate_pointer(rec, ptr)` 校验 `checksum/s_len/per_layer_len` 三者一致，不一致打点 `kv.checksum_mismatch` 并降级（联动 G8）。
- `put` 后：同步写 `KVPointer`（`checksum=rec.checksum`、`s_len`、`per_layer_len`、`obj_key`）。

#### 兼容性/迁移成本对比与推荐

| 维度 | 方案 A（header 内联） | 方案 B（KVPointer 取回） |
|---|---|---|
| 自洽性 | **自洽**：元数据随 blob 走，不依赖控制面 | 依赖 meta 面双写一致，多一个失败/漂移面 |
| 迁移成本 | 低：新增 header 字段 + 读函数 + 改两处 `get` | 中：需后门注入 MetaStore，且 `put` 双写 |
| 与既有代码衔接 | `datasystem_adapter.get/append` 改为读 `read_header` | 复用 `KVPointer`(已存在)/`validate_pointer`(已存在) |
| 风险 | header 版本向后兼容需留 `get` 缺省回退 | payload 与 pointer 可能双写不一致 |

**推荐：A 为主（把有效长度内联到 payload，源头正确），B 为可选交叉校验（防御 pointer 漂移，与 G8 降级联动）。** A 单独即可保证正确，B 单独则仍可能被双写不一致破坏。

#### 落点与衔接

1. `/workspace/onetrans/serving/serialize.py`：`serialize` 增加 `s_len`/`per_layer_len` 参数与 `layers[].len` 字段；新增 `read_header`/`deserialize_with_meta`。
2. `/workspace/onetrans/serving/pipeline.py` 的 `NearlineWorker.ingest`（L62）：调 `serialize(kv.per_layer, s_len=kv.s_len, per_layer_len=kv.per_layer_len)`（`kv` 来自 `encode_s`，已含这些值）。
3. `/workspace/onetrans/serving/datasystem_adapter.py` 的 `get`（L61-77）、`append`（L94-101）：改为读 `read_header(payload)` 拿 `s_len`/每层 `len`；`get` 重建 `UserKVRecord` 时用真实有效长度而非 `shape[1]`。
4. `/workspace/onetrans/serving/local_adapter.py`：无需改（record 内联本就正确）；但 `_rebuild`（L152-166）在非全量场景保持 `rec.per_layer_len` 传入，避免落入 `[k.shape[1] ...]` 推断。

#### 验收标准（数值/正确性）

- 对 `valid_len ∈ {5, 23, 37, 50(满)}` 的左 padding 用户，经 `YuanrongKVStore.put→get`（或 any 无 meta 注入后端）后 `rec.per_layer_len` 逐层等于 nearline 写入值、`rec.s_len == valid_len`。
- 该 record 经 `decode_record` + `score_ns` 与「`local` 后端同输入」逐位一致（max|diff| < 1e-4，沿用 `demo.py` 阈值）。
- `serialize` 旧调用（不传长度）roundtrip 仍通过（向后兼容）。

#### 依赖与风险

- 依赖：无新库；仅 JSON header 字段扩展。
- 风险：header 已发布版本与新增字段的向后兼容——通过 `get(key, 默认回退 shape[1])` 兜底读取旧数据；旧数据默认视为「满宽有效」（与旧语义一致）。

---

### 3.2 G2：datasystem `append` 原子化（CAS / fencing）

#### 目标

消除「读-合并-写」的 TOCTOU 窗口，使并发 append 要么串行化、要么被明确拒绝并触发全量重建，**不丢写**。

#### 接口与数据结构

`AppendResult`（`/workspace/onetrans/serving/kv_store.py` L56-61）已含 `reason` 字段，扩展枚举：

```
reason ∈ {"ok", "missing", "offset_conflict", "cas_conflict"}
```

`DeltaKV`（L69-77）新增可选 `expect_checksum: str = ""`（作为 fencing token / 乐观锁版本；空串＝不校验）。

#### 关键流程

1. **首选（datasystem 提供原子 CAS）**：`put`/`append` 走 `kv().cas(key, expect_checksum, new_payload)`——只有当前对象 checksum（或版本号）等于 `expect_checksum` 才覆盖，否则返回冲突。nearline 侧把上一次 `append` 返回的 `new_checksum` 作为下一次 `expect_checksum`（`expect_checksum` 即 fencing token）。
2. **兜底（datasystem 无 CAS）**：nearline 对同一 user 的写加**分布式排他**（user 级锁 / 按 owner 单写者 + failover 时 fencing token），保证同一 user 的并发写序；一旦 `offset != s_len` 即 `AppendResult(reason="offset_conflict")`，由上层全量重建（幂等，见 `local_adapter.py` L85-98 语义）。
3. **冲突处理**：`reason in {"offset_conflict","cas_conflict"}` → 打点 `kv.version_conflict`（`detailed_design.md` §6.4 已列），nearline 走 `put` 全量重建 + `kv.replace_full`。

#### 落点

- `/workspace/onetrans/serving/datasystem_adapter.py`：`append`（L82-108）改为「条件写」并携带 `expect_checksum`；`put` 同加可选 CAS。
- 桥接 `validate_pointer`（`/workspace/onetrans/serving/meta_store.py` L46-54）：append 成功后再写 pointer（单写者序）。

#### 验收标准

- 并发同 user 双 append（`demo.py:test_append_conflict` 扩展为并发版）：任意时序下最终 `s_len` 等于「按序 append 之和」或「被拒并全量重建后一致」，**不存在静默丢写**；冲突路径打点 `kv.version_conflict` ≥1。

---

### 3.3 G3：路由统一（同 hash64 + 跳变哈希 / PS 统一 Knuth）

#### 目标

使同一 `user_id` 的「KV owner shard」与「处理 worker」落到同一物理位置，最大化数据本地性，并让扩缩容只做最小 remap。

#### 统一方案

- **KV 分片与 worker 分派复用同一个 `Router`**（同一 `hash64` + `JumpConsistentHash`）：
  - `WorkerPool.worker_for`（`/workspace/onetrans/serving/dispatcher.py` L106-107）从 `hash64(key) % num_workers` 改为走注入的 `Router`；`Dispatcher._choose_worker`（L205-211）在 `mode="hash"` 下同步改为 `self.router.route(user_id)`。
  - `ShardedKVStore`（`/workspace/onetrans/serving/sharded.py`）已用 `Router`（jump），保持不变。
- **PS 分片统一 Knuth**：由并行工作项实现（G10），本文不重复。

#### 与既有代码对齐的最小改动

```python
# dispatcher.py 构想（不落地，仅示意）
class WorkerPool:
    def __init__(..., router: Router | None = None):
        self.router = router or Router(num_shards=num_workers)
    def worker_for(self, key: str) -> int:
        return self.router.route(key)          # 替换 hash64(key) % num_workers
class Dispatcher:
    def _choose_worker(self, user_id: str) -> int:
        if self.mode == "hash":
            return self.pool.worker_for(user_id)   # 内部已走 Router
        ...
```

#### `num_workers != num_shards` 时的 locality 权衡

- **首选 `num_workers == num_shards`**（每节点一个 worker + 一个 KV shard，一一共址）。
- 当二者不等，需**显式定义映射**：`worker_id = router.route(user_id) % num_workers`（同一跳变哈希先落到 shard，再折合到 worker）——这样「同 shard 的 user 集合」恒被同一批 worker 覆盖，读仍本地命中；但**避免**「shard 与 worker 分别独立 jump/取模」造成的无法对齐。
- 折衷命名建议：`Router(num_shards)` 为主路由源；worker 分派用 `shard -> worker` 的显式归属表（`worker = shard % num_workers` 或 `shard // (num_shards/num_workers)`），保证确定性与最小迁移。

#### 落点

- `/workspace/onetrans/serving/dispatcher.py`：`WorkerPool` 增加 `router` 注入，`worker_for` 改走 `Router`；`Dispatcher`（`mode="hash"`）不变调用。
- 复用 `/workspace/onetrans/serving/router.py` 的 `hash64`/`JumpConsistentHash`/`Router`（已存在，无需新实现）。

#### 验收标准

- `num_shards == num_workers` 时，对随机 1000 个 user：`shard_of(uid) == worker_for(uid)` 成立（同一跳变哈希，两处对同一 uid 稳定同值）。
- 扩缩容（如 8→9）：`router.route` 的 remap 比例受控（复用 `remap_ratio`，`demo.py` 已断言 <0.2）；不再出现取模法近全量 remap。

---

### 3.4 G4：可靠性四件套（超时 / 重试&幂等 / 熔断限流 / 健康检查优雅停机）

#### 客户端超时

- `Dispatcher.submit`（`/workspace/onetrans/serving/dispatcher.py` L166-190）增加 `deadline`：`Future` 在超时后 `set_exception(TimeoutError)` 并从 `_inflight` 移除；worker 侧如底层支持 cancel（brpc `Controller`），送 cancel 信号。
- `YuanrongKVStore`/`EmbeddingPSClient` 的 `get`/`lookup` 增加 `timeout_ms` 参数透传到 SDK/brpc。
- 落点：`dispatcher.py`、`datasystem_adapter.py`、`embedding_ps_client.py`。

#### 重试 & 幂等

- **读幂等可安全重试**：`get`/`mget`/`lookup` 对 `TimeoutError`/瞬时 I/O 错误做有限次重试（默认 1 次，指数退避）。
- **写幂等**：`put` 以 `checksum` 幂等（同 checksum 去重，`PutResult` 已 carry checksum）；`append` 以 `expect_checksum`（见 3.2）保证重试不重复追加。
- 落点：adapter 层封装，不侵入上层 worker 语义。

#### 熔断 / 限流

- 在 `Dispatcher` 与 adapter 之间加后端级熔断：以滑动窗口错误率（默认 >50% 且 ≥N 次）触发 `open`，`open` 期间快速失败（`CircuitOpen`），半开探测恢复。
- 令牌桶限流于 `submit` 入口（QPS/字节双重阈值），超限即 `OverloadRejected`（已有信号复用）。
- 落点：`dispatcher.py` 新增 `CircuitBreaker`/`RateLimiter`（纯 Python，无新依赖）。

#### 健康检查 / 优雅停机

- `/healthz`：聚合依赖探针（datasystem 可达、PS 可达、队列水位、熔断状态），异常返回 503。
- 优雅停机（drain & wait）：`WorkerPool.stop()`（`dispatcher.py` L94-103）改为「先停止收新请求（标记 `draining`）→ 排空队列并完成 inflight → 再 join」。

#### 验收标准

- 注入慢后端：调用在 `deadline` 内返回 `TimeoutError`，不无限等待；错误率达到阈值后熔断打开、恢复后自愈。
- drain：停机期间已在队请求全部完成、新请求被拒；`inflight()` 收敛到 0。

---

### 3.5 G5：可观测性落地（指标导出 / 结构化日志 / trace）

#### 指标导出（最小改动）

- `_Histogram`（`/workspace/onetrans/serving/metrics.py` L17-29）改为**分桶直方图**（指数桶，如 1ms~10s），`snapshot()` 输出 `_count/_sum` 与各桶；新增 `serve_metrics(port)` 暴露 Prometheus 文本格式（或 OTel 批处理 exporter）。
- `ServingMetrics` 保持 `timing/count/gauge` 三类原语不变（`MetricsSink` 协议不变），仅替换内部存储与新增导出入口。
- 落点：`metrics.py`。

#### 结构化日志（req_id/trace_id 贯穿）

- 定义上下文：`req_id`（入口生成）、`trace_id`（贯穿 Nearline→Online→PS/datasystem）、`user_id`。
- 在 `Dispatcher.submit` 生成 `req_id`；`OnlineWorker.score`/`NearlineWorker.ingest` 打点携带 `req_id/user_id/trace_id`；KV miss/hit、背压拒绝、降级事件走结构化日志 + `metrics.count` 双通道（联动 `detailed_design.md` §6.4 的事件清单）。
- 落点：`dispatcher.py`、`pipeline.py`。

#### 分布式 trace

- 以 OTel span 包裹「路由 → 读 KV → PS 查表 → 打分」四段，父 span 在 `Dispatcher`，子 span 在 `adapter`/`worker`，字段带 `req_id/trace_id`。
- 落点：`pipeline.py`、`datasystem_adapter.py`、`embedding_ps_client.py`。

#### 验收标准

- `prometheus` 可 scrape `/metrics` 且抽样桶正确；日志含 `req_id/trace_id/user_id`；一次请求可跨 Nearline/Online/PS 关联出一条 trace。

---

### 3.6 G6：C++ 热路径移植（分阶段）

#### 分阶段路线

- **M8a 算子级 C++ 移植 + 数值对齐**：优先移植 `encode_s` 与 `score_ns` 的算子级核心（`_project_s`/`_project_ns`、`_apply_ns_ffn`、scaled-dot-product attention、`_s_attn_mask`/`_cross_attn_mask`）为单一 C++ 扩展（或自定义 op），保持与 Python `two_stage.py` 逐层一致。**黄金基准**：`/workspace/onetrans/serving/two_stage.py` 的 `encode_s/score_ns/score_ns_batch` 与 `demo.py:test_equivalence`（断言 `max|diff| < 1e-4`）。移植中逐步收紧到 1e-6。
- **M8b brpc + bthread worker**：以 `deploy/ps/embedding_server.cc` 为参考模板，新增 `nearline_server.cc`/`online_server.cc`（bthread M:N + 每 worker 有界队列 + req_seq 乱序匹配 + 背压），对齐 `dispatcher.py` 的并发语义。C++ 侧读取同一 checkpoint（`weight_loader.py:save_checkpoint` 输出的 `state_dict`）。
- **M8c vLLM 自定义 op（混合参数化层）**：把 `MixedCausalSelfAttention`/`MixedFFN` 的逐 token `W_ns_list`/`networks_ns_list` 落地为 vLLM 自定义层/op（embedding/token-specific 参数），复用其 KV cache 布局/量化能力。

#### 数值对齐步骤（以 Python 为黄金基准）

1. 固化 Python 输出：用 `demo.py` 对固定 seed/输入跑出逐层 K/V 与 logits 快照（作为 CI 金样本）。
2. C++ 每次算子移植后回放同一输入，比对同一 `max|diff|`，先 1e-4 后收紧 1e-6；失败即 `M8a` 未完成。
3. 全链路：C++ `encode_s` 产出 → Python `score_ns` 反推一致性，vice versa，确保两阶段拆分等价于单前向。

#### 验收标准

- M8a：算子级输出与 Python `max|diff| < 1e-6`（`demo.py` 复现）。
- M8b：30 请求并发完成、req_seq 乱序匹配、背压拒绝（对齐 `demo.py:test_dispatcher` 语义）。
- M8c：C++ 混合层权重加载后与 Python 单前向逐位一致。

---

### 3.7 G7：tokenizer + 稀疏 embedding 查表接入 serving 热路径

#### 目标

把「行为流→查表→编码→写 KV」与「特征服务→查表→打分」两段接成完整链路，但**保持 worker 与存储后端解耦（接口注入，不硬编码 remote）**。

#### 设计（前置接线层，不改 worker 核心语义）

- 定义两个**可注入的前置接口**（Protocol，放 `pipeline.py` 或新 `serving/ingest_adapter.py`）：

```python
class STokenizeLookup(Protocol):
    def encode_s(self, seq_features, seq_masks, seq_timestamps=None) -> tuple[Tensor, Tensor]:
        """行为流原始特征 → embed 查表 → OneTransTokenizer.encode_s → (s_emb, s_mask)"""

class NSTokenizeLookup(Protocol):
    def encode_ns(self, ns_groups) -> tuple[Tensor, Tensor]:
        """候选/上下文原始特征 → embed 查表 → OneTransTokenizer.encode_ns → (ns_emb, ns_mask)"""
```

- **落地组合**：`STokenizeLookup/embed → EmbeddingPSClient.lookup`（fabric ①，`/workspace/onetrans/serving/embedding_ps_client.py` L143）+ `OneTransTokenizer.encode_s`（`/workspace/onetrans/nn/tokenizer.py` L135）；NS 侧同理用 `encode_ns`（L143）。
- **worker 侧**：`NearlineWorker.ingest`（`/workspace/onetrans/serving/pipeline.py` L44-55）签名从「收 `s_emb`」扩展为**可收 `s_emb` 或 `s_features`**（重载形式或新方法 `ingest_features`）；`OnlineWorker.score` 同理。默认实现走注入的 `STokenizeLookup`/`NSTokenizeLookup`，**remote/local 由注入的 client 决定**（`EmbeddingPSClient.local()` 或 brpc），worker 不感知。
- **解耦保证**：worker 只依赖 `Protocol`；embed 后端（本地分片表 / 远程 PS）与 tokenizer 通过构造器注入，与 `KVStore` 解耦方式一致（`kv_store.py` 的存储无关契约同理）。

#### 落点

- `/workspace/onetrans/serving/pipeline.py`：新增 Protocol + 前置包装方法（或新 `serving/ingest_adapter.py`，仅当需要多态时才新建文件；优先在 pipeline 内定义）。
- `/workspace/onetrans/nn/tokenizer.py` / `/workspace/onetrans/serving/embedding_ps_client.py`：无需改，仅被组合调用。

#### 验收标准

- 同一组原始特征经「embed 查表 + tokenize + ingest」与「手工预 tokenize + ingest」的 K/V、logits 逐位一致。
- worker 在 `local` 与注入 `remote`（brpc，占位可换真实实现）下行为一致，无硬编码 host。

---

### 3.8 G8：KV miss 降级（两条路径的语义与返回值约定）

#### 路径一：陈旧读 + 打点（针对「读侧校验失败 / 版本漂移」）

- **语义**：`store.get` 返回的 record 与 `KVPointer` 校验不一致（`validate_pointer` 为 False，见 `/workspace/onetrans/serving/meta_store.py` L46-54）时，**不抛异常**，改用**上一份仍可用的陈旧 K/V**（若 datasystem 具备版本化/多代读），并打点 `kv.stale_read` + `offline.kv_on_hit_score_delta`（`detailed_design.md` §6.5 已列）供离线评估陈旧对分数的影响。
- **返回值约定**：`OnlineWorker.score` 返回 `(logits, HitStatus)`，`HitStatus ∈ {fresh, stale}`；`score_batch` 返回 `list[HitStatus]` 对齐 batch。无陈旧副本时退化到路径二。

#### 路径二：空 KV 快速返回（针对「KV 真缺 / 冷启动」）

- **语义**：`store.get` 返回 `None` 时，**不再 `raise KeyError`**（移除 `/workspace/onetrans/serving/pipeline.py` L103-105 的 `raise`、L127-130 的同理），改为快速返回**确定的占位/兜底结果**并打点 `kv.miss`。
- **返回值约定**：`score` 返回 `(全零 logits [M, T], HitStatus.miss)`；`score_batch` 对 miss 项填全零行（保持输出形状 `[B, T]` 与 batch 展平顺序，避免破坏下游排序）。上层按 `HitStatus`/全零判定「不可用」，可走冷启动兜底（如 seed 初始化、先编码再打分）。
- **打点**：`kv.miss`（已有，`pipeline.py` L104/L128 已 count，仅去掉 raise）。

#### 落点

- `/workspace/onetrans/serving/pipeline.py` 的 `OnlineWorker.score`/`score_batch`。

#### 验收标准

- miss 用户（冷启动）返回全零且 `kv.miss`+1，**不抛异常**；批量打分中 miss 与 hit 混查仍返回张量 `[B, T]` 且顺序对齐。

---

### 3.9 G9：服务发现 / 模型版本注册中心

#### 目标

消除 host/port 硬编码（`127.0.0.1:8000`/`127.0.0.1:31501`），提供「model_version → checkpoint 路径 / PS 表版本 / 灰度开关」的版本化与发现能力。

#### 接口与数据结构（轻量，Control 面）

```python
class ServiceRegistry(Protocol):
    def resolve(self, name: str, model_version: str) -> list[Endpoint]: ...   # 服务发现
    def get_model(self, model_version: str) -> ModelRelease: ...              # 版本注册
    def watch(self, name: str) -> Iterator[list[Endpoint]]: ...               # 变更订阅（可选）

@dataclass
class ModelRelease:
    model_version: str
    checkpoint_dir: str        # weight_loader.load_backbone 的 checkpoint_dir
    checkpoint_file: str       # 缺省用 seed
    ps_table: str              # PS table 名（联动 G11 多表）
    target_weight: float       # 灰度权重 0..1
```

#### 流程

- 启动：`load_backbone(model_version, ...)`（`/workspace/onetrans/serving/weight_loader.py` L51-86）的 `checkpoint_dir` 从 registry 读取而非硬编码。
- `EmbeddingPSClient`/`YuanrongKVStore` 从 `resolve()` 拿 `Endpoint(host,port)`，替换默认 `127.0.0.1`；`watch` 用于滚动重启/摘除感知。
- 灰度：`ModelRelease.target_weight` 按 user 哈希权重分桶到不同 `model_version`。

#### 落点

- 新增 `/workspace/onetrans/serving/registry.py`（若需多实现）或并入 `meta_store.py`；`embedding_ps_client.py`/`datasystem_adapter.py`/`weight_loader.py` 的构造器接受 `registry` 注入，默认回退本地单实例（保持现有 `demo.py` 不依赖外部注册中心即可运行）。

#### 验收标准

- 无外部注册中心时（本地默认）`demo.py` 全量通过；注入 registry 后端点/版本可动态变化且不需改代码/重启即生效。

---

## 第四部分：分阶段落地路线图与验收

### 4.1 里程碑（继承 `detailed_design.md` §8，新增 M5~M8）

| 里程碑 | 内容 | 出口 / 验收标准 | 依赖 |
|---|---|---|---|
| M0 | pyramid 方向修正 + 单卡数值基准 | 单前向 vs 两阶段 max\|diff\| < ε（已达成） | — |
| M1 | `KVStore` 接口 + 本地 adapter + 序列化 | roundtrip 一致性（已达成） | — |
| M2 | datasystem adapter（KV） | 集群读写基准（部分；待 G1 修正元数据） | M1 |
| **M5（正确性收口）** | G1 元数据固化 + G2 append 原子 + G3 路由统一 + G8 miss 降级 | **已达成**：①header 固化 `s_len`/`per_layer_len`，roundtrip 校验逐层一致；②并发 append 不丢写（offset/cas 冲突均拒绝）；③`worker_for(uid)==Router.route(uid)`（同 shard 数），扩缩容 remap 受控；④miss 返回全零不抛异常（单/批一致） | M2 |
| M3 | 指标埋点 + 端到端负载（含 G5 最小导出） | §6 全指标可采、分桶直方图 + `/metrics` | M5 |
| **M6（可靠 + 观测）** | G4 四件套 + G5 结构化日志/trace | 超时/重试/熔断/健康检查/优雅停机齐备；req_id/trace_id 贯穿 | M5 |
| **M7（热路径接线）** | G7 tokenizer+embedding 接线 + G9 服务发现/注册 | 「行为流→查表→编码→写 KV→读 KV→打分」端到端闭环；host/port 无硬编码，可灰度 | M5 |
| **M8（C++ 移植）** | G6 分阶段：M8a 算子级 / M8b brpc worker / M8c vLLM op | M8a 数值对齐 1e-6；M8b 并发/乱序/背压对齐；M8c 混合层权重逐位一致 | M7 |
| M4 | DeepFM/DCNv2 等约束公平对比 | Pareto 前沿（依赖 M3/M7 同 seam） | M3、M7 |

### 4.2 本次并行「已完成 vs 待做」清单

**本次并行已完成（本会话 C++ 数据面客户端工作项，详见代码注释，本文不重复设计）**

- [x] G10：PS 跨语言分片哈希统一到 Knuth（`/workspace/deploy/ps/embedding_server.cc` + `/workspace/onetrans/serving/embedding_ps_client.py`）。
- [x] G11：C++ PS 单表 → 多表（`table -> ShardedEmbeddingTable`），支持多模型版本/灰度。

**M5 正确性收口已完成（本会话，`demo.py` 端到端校验通过）**

- [x] G1：序列化 header 固化 `s_len`/`per_layer_len`（`serialize.py` + `pipeline.py`/`datasystem_adapter.py`/`local_adapter.py` 适配）。
- [x] G2：`DeltaKV.expect_checksum` CAS fencing（local/datasystem 双侧 `append`，`cas_conflict` 拒绝）。
- [x] G3：`WorkerPool.worker_for` 复用 `Router`（jump 哈希），worker 分派与 KV 分片统一。
- [x] G8：KV miss 降级为全零 logits + `kv.miss` 打点（单请求/攒批一致，不再 `raise KeyError`）。

**待做（本/后续会话）**

- G4 / G5（M6 可靠 + 观测）
- G7 / G9（M7 热路径接线）
- G6（M8 C++ 移植，分 M8a→M8c）
- G12 ~ G14（P2：redis/HBM 直通、测试框架/CI、局部性能）

> 本文档按「设计先行」维护：G1~G3/G8/G10/G11 的落地实现见对应代码文件与 `implementation_status.md` §2；`.cc/.h/.py` 的具体改动不在本文展开。
