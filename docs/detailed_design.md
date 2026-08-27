# 序列 Transformer 精排系统 · 端到端详细设计

> 版本：v1.0
> 文档类别：**② 端到端设计**——**只依据设计意图**描述系统应该是什么样：业务诉求、需求分析、输入输出、功能设计、实现分析与实现设计。
> 边界约定：本文**不含**实现进度、已实现/未实现标注、差距与里程碑（一律见 [implementation_status.md](./implementation_status.md) / [gap_analysis.md](./gap_analysis.md)）；模型结构/训练/等价性论证等算法层内容见 [model_design.md](./model_design.md)。
> 上游：[端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6）
> 交付依赖：本文所有组件均为**目标设计**，落地顺序与验收状态以现状文档为准。

---

## 0. 文档结构与阅读导航

**文档三分体系**（严格按内容类别划分，互不混杂）：

| 类别 | 文档 | 内容 |
|---|---|---|
| ① 模型层 | [model_design.md](./model_design.md) | 模型结构、计算逻辑、训练、两阶段等价性、正确性验证方法学 |
| ② 端到端设计 | [e2e_design_spec.md](./e2e_design_spec.md)（概要）+ **本文**（详细） | 只依据设计意图的系统设计：架构/数据流/线程模型/契约 |
| ③ 现状 & 差距 | [implementation_status.md](./implementation_status.md) + [gap_analysis.md](./gap_analysis.md) | 已实现/未实现、实测结果、差距分级与路线图 |

| 读者 | 关注章节 |
|---|---|
| 产品/业务方 | §1 业务诉求、§2 需求分析、§3 输入输出 |
| 算法工程师 | model_design.md（全文）、§4.2 两阶段系统映射 |
| 存储工程师 | §4.3 KV 接口契约、§7.5 关键数据结构、§7.6 序列化布局 |
| 后端/并发工程师 | **§7.1 生产架构（主视图）**、**§7.4 线程与并发模型**（含具体请求端到端走读） |
| 交付负责人 | model_design.md §6（验证矩阵）+ ③ 类文档（验收状态） |

约定符号：`D=d_model`，`H=num_heads`，`d=head_dim=D/H`，`L=num_blocks`，`Ns=ns_tokens_num`，`S_len` 有效序列长度，`M` 单请求候选数，`B` batch 维，`ΔL` 增量行为条数。

---

## 1. 业务诉求

### 1.1 业务背景

推荐系统精排（ranking）位于召回/粗排之后，对每个请求的候选集（数百个 item）逐一预估 pCTR/pCVR 等目标分数，直接决定最终展示次序，是业务收益的核心杠杆。传统精排（DeepFM/DCN 等「宽而深」结构）对用户行为历史的建模依赖池化或简单注意力，难以捕捉**长程、时序**的依赖；序列 Transformer 精排（OneTrans 类）把用户行为序列视作 token 序列，用自注意力直接建模序列内依赖，再用候选 token 与历史做交叉注意力打分，代表算法侧的升级方向。

但该类算法引入一个工程难题：**每请求都要对用户历史重新做 prefill 级别的编码**。若用户序列长 `S_len=100`、模型 `L=6` 层，则每次打分前都要重复计算约 `100×6` 个 token 位置的自注意力编码——而同一用户在短时间内会被反复请求（一次刷新页面对同一 user 发起多次打分），历史编码结果高度可复用却被反复重算，算力与带宽随序列长度线性膨胀。

### 1.2 业务痛点（现状问题）

| # | 痛点 | 业务后果 |
|---|---|---|
| P-1 | 用户历史每请求重复编码 | 在线打分时延随序列长度线性劣化，长序列用户打分 p99 超出 SLO |
| P-2 | 序列模型算力开销大 | 同等 GPU 预算下 QPS 承载能力显著低于传统精排，覆盖不住流量高峰 |
| P-3 | 新用户/冷启动无历史 | KV 缺失若硬失败，错误率被无谓抬高，影响灰度与放量 |
| P-4 | 行为实时性要求 | 用户刚产生的新行为若不能反映到下一次打分，转化窗口内损失收益 |
| P-5 | 模型迭代频繁 | 新版本灰度需新旧版本并存，权重/embedding 表需版本化管理 |
| P-6 | 线上排障困难 | 出现「某用户分数异常」时若无链路可观测能力，无法归因定界 |

### 1.3 业务价值主张

把 OneTrans 类算法的**两阶段结构（用户侧摊销编码 + 候选侧交叉打分）**与 LLM 推理的 **prefill/decode 拆分** 对应起来：用户历史编码结果（逐层 K/V）作为可复用状态**摊销**到该用户的多次打分请求上，用**分布式 KV Cache + 数据本地化**承接。业务价值：

1. **时延**：在线阶段只做「读缓存 + 交叉注意力」，把 O(S_len×L) 的编码从请求关键路径上移除；
2. **吞吐**：M 个候选整批并行（非自回归），交叉注意力阶段高密度算力利用；
3. **实时性**：行为流近线增量更新用户 K/V（append），分钟级反映到打分；
4. **韧性**：KV miss 降级为确定性的零分兜底（不硬失败），灰度/冷启动窗口错误率可控；
5. **演进性**：模型版本/稀疏表版本化 + 一致性哈希路由，支撑灰度发布与水平扩缩容。

### 1.4 系统定位与边界

- **定位**：面向推荐精排场景的**序列 Transformer 在线/近线推理服务系统**，研究「集群规模下该类算法的工程落地」，不追求训练最优模型。
- **在范围内**：Nearline（Stage I 用户历史编码）、Online（Stage II 候选交叉打分）、分布式 UserKV 存储、稀疏 Embedding 数据面（PS）、端到端数据流、指标采集、并发与路由。
- **不在范围内**：召回/粗排、在线 A/B 平台、训练参数服务器与异步训练、广告计费/控制面。

---

## 2. 需求分析

### 2.1 功能性需求（FR）

| 编号 | 需求 | 说明 | 实现落点 |
|---|---|---|---|
| FR-1 | 用户历史近线编码 | 消费行为流，对每个用户做 S 侧逐层编码，产出并存储逐层 K/V | `NearlineWorker.ingest` + `TwoStageRunner.encode_s` |
| FR-2 | UserKV 存储与读取 | 以 `(model_version, user_id)` 为主键，存储/读取整套逐层 K/V，存储后端可替换 | `KVStore` 协议 + `LocalKVStore`/`YuanrongKVStore` |
| FR-3 | 候选在线交叉打分 | 读用户 K/V，对 M 个候选做 NS 侧交叉注意力，输出 `[M, T]` 分数 | `OnlineWorker.score` + `TwoStageRunner.score_ns` |
| FR-4 | 动态攒批 | 多请求按「满批或超时」聚合为一次前向，提升吞吐、时延有界 | `BatchScheduler` + `score_ns_batch` |
| FR-5 | 增量 append | 用户新增 ΔL 条行为时，按 offset 追加 K/V，冲突必须显式拒绝（不静默丢写） | `KVStore.append` + `DeltaKV`（offset + CAS 双校验） |
| FR-6 | KV miss 降级 | 缓存缺失/过期时返回全零合法输出并打点，不抛异常 | `OnlineWorker.score/score_batch` |
| FR-7 | 一致性哈希路由 | user → owner worker/shard 稳定映射，数据本地化 + 最小 remap | `Router`（jump/ring）+ `ShardedKVStore` + `WorkerPool.worker_for` |
| FR-8 | 元数据/版本失效 | KV 指针（checksum/长度/时间戳）+ TTL 惰性过期 | `KVPointer` + `LocalMetaStore` + `validate_pointer` |
| FR-9 | 稀疏 Embedding 查表 | 独立 PS（多表、分片、版本化），未命中确定性 seed 兜底 | `EmbeddingPSClient` + `deploy/ps/embedding_server.cc` |
| FR-10 | 权重版本化加载 | checkpoint 优先、seed 兜底，按 model_version 装载 | `weight_loader.load_backbone` |
| FR-11 | 指标采集 | 时延直方图/计数/仪表，命名 `组件.阶段.子操作` | `ServingMetrics`（timing/count/gauge） |
| FR-12 | 并发调度 | N worker 独立有界队列 + req_seq 异步匹配 + 背压 | `Dispatcher` + `WorkerPool` |

### 2.2 非功能性需求（NFR）

| 编号 | 维度 | 指标/要求 | 验证方式 |
|---|---|---|---|
| NFR-1 | 正确性 | 两阶段拼接与单前向**数值等价**，max\|diff\| < 1e-4（容差分级见 model_design §6.2） | 等价性断言（验证矩阵 V1） |
| NFR-2 | 正确性 | KV 元数据（s_len/per_layer_len）跨后端不丢失 | 序列化 roundtrip 断言（左 padding 用户） |
| NFR-3 | 一致性 | 并发 append 不静默丢写；冲突显式拒绝 | append 冲突断言（offset_conflict + cas_conflict） |
| NFR-4 | 时延 | 打分 p50/p99/p999 可测、可分解到阶段（kv_get / encode_stage2） | `timing` 埋点 |
| NFR-5 | 吞吐 | 攒批窗口内满批聚合；批量与逐条打分数值一致 | 批量等价断言（验证矩阵 V2） |
| NFR-6 | 可用性 | KV miss 不硬失败（全零降级）；权重缺失 seed 兜底；队列满快速拒绝 | miss/权重/背压用例（V6） |
| NFR-7 | 可扩展 | 节点增减 remap 比例 O(k/n)（jump 哈希），非全量迁移 | 路由 remap 断言 |
| NFR-8 | 数据本地性 | 同一 user 的 KV shard 与处理 worker 同桶 | `worker_for(uid) == Router.route(uid)` 断言 |
| NFR-9 | 可观测 | 每阶段埋点：nearline.encode_stage1 / online.kv_get / online.encode_stage2 等 | 指标快照用例 |
| NFR-10 | 零拷贝 | 序列化读侧 `frombuffer` 视图；mmap 后端读侧免二次拷贝 | 零拷贝断言 |

### 2.3 约束与假设

- **单卡=单任务**：单模型规模小（单 GPU 可跑），但用户 KV 必须分布式（D1 决策）。
- **KV 存储一致性仅 Causal/PRAM**：datasystem 类后端不提供线性一致；在线读侧靠 checksum/版本校验兜底。
- **Nearline 按 user 单写者**：默认部署下同一 user 的写入串行（一致性哈希单 owner）；CAS fencing 为打破该假设时的安全网。
- **第二阶段非自回归**：M 候选整批并行，无逐 token 循环依赖——这决定了线程模型与 LLM decode 服务不同（无需 PD 分池调度）。
- **左 padding 语义**：有效 token 靠序列尾部（最新行为在尾），头部为 padding——所有掩码、有效长度、pyramid 剪裁方向都由此推导。

### 2.4 需求 → 组件追踪矩阵

| 需求 | 主责组件 | 协作组件 |
|---|---|---|
| FR-1 | pipeline.NearlineWorker | two_stage / serialize / kv_store |
| FR-2 | kv_store.KVStore 协议 | local_adapter / datasystem_adapter / sharded |
| FR-3 | pipeline.OnlineWorker | two_stage / kv_store |
| FR-4 | pipeline.BatchScheduler | two_stage.score_ns_batch |
| FR-5 | kv_store.append | local_adapter / datasystem_adapter |
| FR-6 | pipeline.OnlineWorker | metrics |
| FR-7 | router.Router | sharded.ShardedKVStore / dispatcher.WorkerPool |
| FR-8 | meta_store.LocalMetaStore | kv_store.UserKVRecord |
| FR-9 | embedding_ps_client | deploy/ps（C++ brpc） |
| FR-10 | weight_loader | models.OneTrans |
| FR-11 | metrics.ServingMetrics | pipeline / dispatcher |
| FR-12 | dispatcher.Dispatcher | dispatcher.WorkerPool |

---

## 3. 输入输出

### 3.1 系统级输入输出

```
输入                                    本系统                                   输出
────                                    ────                                    ────
E1 行为事件流                ┌─────────────────────────────┐                F1 候选分数 [M, T]
   {user_id, ΔL 个行为,        │  Nearline: tokenize → Stage I │── UserKV ──▶  (pCTR/pCVR logits)
    timestamps}               │  编码 → put/append KV         │   (fabric ②)     ▲
                              └─────────────────────────────┘                   │
E2 在线打分请求                ┌─────────────────────────────┐                   │
   {user_id, model_version,    │  Online: 读 KV → Stage II     │─────────────────┘
    M 个候选特征}              │  交叉注意力 → head            │
                              └─────────────────────────────┘
E3 候选/用户/上下文特征 ID ──▶ fabric ① 稀疏 Embedding PS（查表）──▶ embedding 向量
E4 模型发布（checkpoint/表） ─▶ fabric ③ 元数据/版本面（注册/失效）
```

### 3.2 组件接口与 I/O 契约总表

| 组件 | 输入 | 输出 | 失败语义 |
|---|---|---|---|
| `STokenizer.encode_s` | `seq_features[]`（各行为类型特征）、`seq_masks[]`、`seq_timestamps[]` | `(s_emb [1,S0,D], s_mask [1,S0])` | 无（确定性投影 + 时间戳排序合并） |
| `NSGroupWiseTokenizer.encode_ns` | `ns_groups[]`（候选/上下文/画像特征组） | `(ns_emb [M,Ns,D], ns_mask [M,Ns])` | 无 |
| `TwoStageRunner.encode_s` | `s_emb [1,S0,D]`、`s_mask [1,S0]` | `UserKV{per_layer[(K,V)]×L, per_layer_len, s_len}` | `ValueError`（B≠1） |
| `TwoStageRunner.score_ns` | `UserKV`、`ns_emb [M,Ns,D]` | `logits [M,T]` | 无 |
| `TwoStageRunner.score_ns_batch` | `kvs: list[UserKV]`（B 个）、`ns_emb [B,Ns,D]` | `logits [B,T]` | `ValueError`（B 不匹配） |
| `serialize` | `per_layer`、`s_len`、`per_layer_len` | `bytes`（payload） | `AssertionError`（K/V dtype 不一致） |
| `deserialize(_with_meta)` | payload（bytes/bytearray/memoryview/mmap） | 逐层 `(K,V)` 视图（+ 元数据） | `ValueError`（魔数不符） |
| `KVStore.put` | `UserKVRecord` | `PutResult{accepted, version, checksum}` | 按 reason 拒绝 |
| `KVStore.get/mget` | `KVKey(±layers)` | `UserKVRecord | None` | miss 返回 None |
| `KVStore.append` | `DeltaKV{offset, delta_len, tensors, expect_checksum}` | `AppendResult{accepted, new_s_len, checksum, reason}` | missing / offset_conflict / cas_conflict / layer_mismatch |
| `Dispatcher.submit` | `user_id`、payload、timeout | `Future[Response]` | `OverloadRejected`（队列满，异步异常） |
| `BatchScheduler.submit/next_batch` | `ScoreRequest` | `list[ScoreRequest]`（≥1，满批或超时） | 无 |
| `EmbeddingPSClient.lookup` | 表名、`ids [N]` | `weights [N,dim]` | miss → 0 向量 / seed 兜底 |
| `load_backbone` | `model_version`、checkpoint_dir、seed | `(OneTrans, source)` | 缺失/损坏 → seed 兜底，source="seed" |

### 3.3 张量契约（T1~T10）

布局统一 `bshd = [b, S, H, d]`，进入 `scaled_dot_product_attention` 前转置为 `bhsd`。默认 dtype fp16（demo 小配置 D=128, H=4, L=4, max_seq=50, Ns=8）。

| # | 边界 | 张量 | 形状 | 生产者 → 消费者 |
|---|---|---|---|---|
| T1 | Nearline 输入 | `S_emb`（token+pos+type emb，RMSNorm 后） | `[1, S0, D]` | S tokenizer → Stage I Block0 |
| T2/T3 | Stage I 逐层缓存 | `K_s^l / V_s^l` | `[1, S_l, H, d]`，`S_l=dims[l]` | Stage I Block l → KV Store |
| T4 | Stage I 中间态 | `S_hidden^l` | `[1, S_l, D]` | Block l → Block l+1（不缓存） |
| T5 | Online 输入 | `NS_emb` | `[M, Ns, D]` | NS tokenizer → Stage II |
| T6 | Stage II Q | `Q_ns^l` | `[M, Ns, H, d]` | NS 逐 token 投影 → 交叉注意力 |
| T7 | Stage II NS 自身 K/V | `K_ns^l / V_ns^l` | `[M, Ns, H, d]` | NS 逐 token 投影 → 交叉注意力 |
| T8 | 交叉注意力 K/V | `K^l = [K_s^l ∥ K_ns^l]` | `[M, S_l+Ns, H, d]` | KV 缓存 + 在线 NS 拼接 |
| T9 | Stage II 输出 | `NS_hidden^L` | `[M, Ns, D]` | Block L → head |
| T10 | 最终分数 | `logits` | `[M, T]`（T=2：pCTR/pCVR） | head → 上游排序 |
| — | 批量打分 | `score_ns_batch` 输入 | `kvs: list[UserKV] × B`、`ns_emb [B,Ns,D]` | BatchScheduler → Stage II |
| — | 增量 append | `ΔK_s^l / ΔV_s^l` | `[1, ΔL, H, d] × L 层` | Stage I（增量）→ KV append |

---

## 4. 功能设计

### 4.1 功能分解（FBD）

```
序列 Transformer 精排系统
├── F1 模型与算法
│   ├── F1.1 OneTrans backbone（L×CoreOneTransBlock：RMSNorm+MixedAttn+MixedFFN）
│   ├── F1.2 混合参数化（S 共享投影 / NS 逐 token 独立投影）
│   ├── F1.3 金字塔降层（逐层宽度 dims[l] 递减，尾部保留最新）
│   └── F1.4 打分头（末 Ns token 池化 或 CLS token → linear [D,T]）
├── F2 两阶段推理引擎（TwoStageRunner）
│   ├── F2.1 encode_s：S 侧逐层编码 + K/V 缓存（prefill 类比）
│   ├── F2.2 score_ns：读缓存交叉注意力打分（decode 类比，并行非自回归）
│   └── F2.3 score_ns_batch：批量（user,候选）对一次前向
├── F3 存储数据面
│   ├── F3.1 KVStore 协议（put/get/mget/append/delete/ttl/prefetch）
│   ├── F3.2 序列化（零拷贝 read/write + 元数据固化）
│   ├── F3.3 后端 adapter（LocalKVStore[mmap] / YuanrongKVStore / ShardedKVStore）
│   ├── F3.4 元数据面（KVPointer / TTL 惰性过期 / 校验）
│   └── F3.5 稀疏 PS 数据面（Knuth 分片 + 多表 + seed 兜底）
├── F4 服务编排
│   ├── F4.1 NearlineWorker（ingest：编码→序列化→put）
│   ├── F4.2 OnlineWorker（score/score_batch：get→解码→打分，miss 降级）
│   ├── F4.3 BatchScheduler（FIFO 攒批，满批或超时）
│   └── F4.4 Dispatcher/WorkerPool（路由派发 + req_seq 匹配 + 背压）
└── F5 横切
    ├── F5.1 一致性哈希路由（jump/ring）
    ├── F5.2 指标采集（timing/count/gauge）
    └── F5.3 权重版本化加载（checkpoint/seed）
```

### 4.2 核心算法与两阶段系统映射（算法详见 [model_design.md](./model_design.md)）

模型结构（OneTrans：L×CoreOneTransBlock、混合参数化、金字塔降层、tokenizer）与两阶段等价性论证（不变量 I1~I3）属模型层内容，**统一见 [model_design.md](./model_design.md)**。此处只保留驱动 serving 设计的系统映射：

| 维度 | LLM 推理 | 本系统 |
|---|---|---|
| 第一阶段 | prefill（全 prompt） | **Nearline（Stage I）**：S 侧逐层编码 + 缓存每层 `(K_s^l, V_s^l)` |
| 第二阶段 | decode（逐 token 自回归） | **Online（Stage II）**：NS 逐层交叉注意力打分（**并行非自回归**） |
| 共享态 | KV cache | **UserKV**（S 侧逐层 K/V，fabric ②） |
| 摊销收益 | 避免重复 prefill | 用户历史编码一次、打分多次摊销 |

**等价性成立 ⟹ 工程拆分合法**：S 段 K/V 与 NS 无关（因果隔离）且可预计算，故可把 O(S_len×L) 编码移出在线关键路径——这是本系统一切存储/路由/并发设计的算法前提（论证见 model_design §4）。

**与 LLM 推理服务的关键差异**：第二阶段无 sequential 依赖，M 候选整批并行，瓶颈是「KV 读取带宽 + 交叉注意力算力」而非「逐 token 时延」；故**无需 PD 分池**，而是 KV shard 与 online worker **同节点共存**（一致性哈希同桶）以最大化数据本地性。

**对工程契约的算法约束**（来自模型形态，工程必须遵守）：
1. 左 padding：有效 token 在尾部 ⟹ 掩码/有效长度/剪裁方向全部由此推导；
2. 逐层 `S_l = dims[l]` 不同（pyramid）⟹ 序列化必须携带**每层有效长度**元数据（§4.3/§7.6 的契约动因）；
3. NS 逐 token 独立投影 ⟹ 在线算子是「逐 token 小 GEMM 序列」，批量化设计（§7.3.3）必须保持形状一致。

### 4.3 KV 存储接口契约

`KVStore(Protocol)`（[kv_store.py](file:///workspace/onetrans/serving/kv_store.py)，`@runtime_checkable`）：`connect/close/put/get/mget/append/delete/ttl/prefetch`。

| 操作 | 输入 | 输出 | 幂等性 | 一致性 |
|---|---|---|---|---|
| `put` | 完整 `UserKVRecord` | `PutResult{accepted, version, checksum}` | 幂等（同 checksum 覆盖） | Causal |
| `get` | `KVKey` + 可选 `layers` | `UserKVRecord｜None` | 只读 | Causal/PRAM |
| `mget` | `list[KVKey]` | 对齐列表（缺位 None） | 只读 | Causal/PRAM |
| `append` | `DeltaKV{key, base_version, offset, delta_len, tensors, expect_checksum}` | `AppendResult{accepted, new_s_len, checksum, reason}` | 非幂等（offset+CAS 校验） | Causal |
| `delete` | `list[KVKey]` | `DeleteResult{deleted}` | 幂等 | Causal |
| `ttl` | `KVKey, seconds` | — | — | — |
| `prefetch` | keys + dest(hbm/dram) | 零拷贝引用（ObjRef） | — | — |

**append 冲突语义**（乐观并发 + CAS fencing）：服务端依次校验 ① `offset == rec.s_len`（否则 `offset_conflict`）② `expect_checksum` 非空时 `== rec.checksum`（否则 `cas_conflict`）③ 层数匹配（否则 `layer_mismatch`）。冲突显式拒绝，由上层全量 `put` 重建兜底，**绝不静默丢写**。

**key 规范**：`kv:{b64(model_version)}:{b64(user_id)}`（url-safe base64 去 padding，满足 datasystem 字符集约束）；元数据键 `meta:{同上}`。

### 4.4 关键功能规格

- **ingest（FR-1）**：`s_emb/s_mask → encode_s → serialize(s_len, per_layer_len) → UserKVRecord → store.put`；埋点 `nearline.encode_stage1`/`nearline.append_kv`/`nearline.events_ingested`/`kv.obj_cnt`。全量 prefill + put 为基线通道，增量 append 为增强通道（语义见 §4.3）。
- **score（FR-3/6）**：`get → (None? zeros+kv.miss : decode_record → score_ns)`；埋点 `online.kv_get`/`online.encode_stage2`/`kv.hit`/`kv.miss`/`online.qps`/`online.candidate_throughput`。
- **score_batch（FR-4/6）**：一次 `mget` + miss/hit 分流（miss 候选全零、hit 候选展平）+ 一次 `score_ns_batch` 前向 + 按展平顺序还原 `[ΣM, T]`。
- **路由（FR-7）**：`JumpConsistentHash`（Lamping-Veach，O(ln n) 无状态，固定桶数最小 remap）为主；`RingHash`（ketama，128 虚拟节点/物理节点）供动态增删。`hash64 = sha256(key)[:8]` 大端（跨进程稳定，不用 Python 内置 `hash`）。
- **PS（FR-9）**：分片哈希 Knuth 乘法 `(id * 0x9E3779B97F4A7C15) mod 2^64 mod num_shards`，Python 与 C++ 逐位对齐（跨语言契约）；多表注册（`table → ShardedEmbeddingTable`，每表独立 version）；miss → 0 向量 / 客户端确定性 seed 兜底。

### 4.5 稀疏 Embedding PS 数据面契约

**设计要点**：

1. **按 id 稳定哈希分片（N 分片，每分片一把锁）**：读写细粒度并发，无全局锁；分片哈希以 Knuth 乘法为**唯一标准**（跨语言逐位对齐，R-5）。
2. **表版本 `version` 随写递增**：供「权重版本化加载 / 失效校验」；多表（多模型版本）以 `table` 字段区分（R-6）。
3. **未命中回 0 向量，客户端 seed 兜底哈希嵌入重建**：权重版本化最差路径。

**wire 契约（`embedding_service.proto`）**：

| RPC | 请求字段 | 响应字段 |
|---|---|---|
| `Lookup` | `table`（表名/模型版本）、`ids[]`（int64 特征 id）、`dim` | `weights[dim]`（float）、`version`（表版本）、`shard_id` |
| `BatchLookup` | `table`、`ids[][]` | 同上按批对齐 |

**部署形态**：独立 PS 服务（C++ brpc + bthread），Nearline / Online 两池共享查表（稀疏表可超单机内存，独立服务化避免每实例各持一份）；服务端内部 `TableRegistry: table_name → ShardedEmbeddingTable`（懒建 + 淘汰，每表独立 version_），`DoLookup` 按 `req.table()` 路由。

---

## 5. 实现分析

### 5.1 技术选型对比

| 决策点 | 候选 | 选择 | 理由 |
|---|---|---|---|
| 参照语言 | Python / C++ / Rust | **Python 参照 + C++ 生产** | 先以 Python 固化数值基准（黄金基准），热路径后移 brpc+bthread（确定性低时延、M:N） |
| 序列化 | pickle / safetensors / 自定义 | **自定义 header + raw bytes** | 禁止 pickle 任意对象（安全 + 跨语言）；header 承载 dtype/shape/有效长度，raw 部分可 frombuffer 零拷贝 |
| KV 后端 | Redis / 自研 / datasystem | **KVStore 协议 + 多 adapter** | 存储无关：LocalKVStore（单机基准/mmap 零拷贝）、YuanrongKVStore（生产，HBM/DRAM 多级）、redis（预留） |
| 路由 | 取模 / jump / ring | **jump 为主 + ring 备** | 取模扩缩容全量 remap；jump 最小 remap 且无状态；ring 支持节点集合任意变化 |
| 并发模型 | 全局队列+单锁 / 每 worker 独立队列 | **N 独立有界队列** | 消全局锁：单锁会串行化所有提交（队头阻塞），独立队列 + hash 派发才能横向扩展 |
| 异步匹配 | 同步阻塞 / 回调 / Future | **Future + req_seq** | 支持乱序完成、超时取消、异常传播；与 brpc Controller/done 回调语义同构 |
| 攒批 | 固定批 / 请求级 | **FIFO + 满批或超时窗口** | 时延有界（max_wait 兜底）+ 吞吐优先（满批聚合） |

### 5.2 关键实现风险与设计对策

| # | 风险 | 本质 | 设计对策 |
|---|---|---|---|
| R-1 | 元数据隐性依赖 | 序列化 header 只存 dtype/shape 时，跨后端读写丢失有效长度 → 左 padding 用户在线掩码错误且**静默** | header 显式固化 `s_len`/`layers[].len` 为契约；旧数据回退满宽（向后兼容）；roundtrip 断言入验证矩阵 |
| R-2 | append TOCTOU | 「读-合并-写」三段式无原子栅栏：并发 append 后写覆盖先写，**丢写无告警** | `expect_checksum` CAS fencing，冲突显式拒绝；后端具备原子 CAS 时直通 |
| R-3 | 路由不一致 | KV 分片与 worker 派发哈希不同 → 同一 user 的 KV 与 worker 落不同节点，跨节点读 + 扩缩容全量 remap | worker 派发复用同一 `Router`（jump），与 KV 分片同桶；`num_shards == num_workers` |
| R-4 | miss 硬失败 | KV miss（冷启动/TTL 过期/分片迁移）为常态事件，硬失败抬高错误率 | 全零 logits + `kv.miss` 打点（单/批一致），上层判定不可用走兜底 |
| R-5 | 跨语言哈希不等价 | Python 与 C++ 分片哈希不同 → PS 读写两侧分片错位 | 分片哈希以 Knuth 乘法为唯一标准，两侧逐位对齐 + 黄金值校验 |
| R-6 | 单表 PS | PS 忽略 table 字段 → 无法多模型版本灰度 | `TableRegistry` 多表注册 + 每表独立 version |
| R-7 | 数值漂移 | 两阶段与单前向不等价（移植/替换算子时） | 等价性断言每次回归；容差分级（model_design §6.2） |

### 5.3 依赖分析

- 运行时：`torch`（CPU 可跑，GPU 可选）、标准库（queue/threading/mmap/ctypes/struct/json/hashlib）。
- 外部系统（按后端可选）：yuanrong datasystem SDK（`YuanrongKVStore`，惰性 import，缺失不 crash 其余路径）；brpc+protobuf（`deploy/ps`，bazel 构建）。
- 无数据管线/wandb/cuda 强依赖：`demo.py` 全链路可在纯 CPU 环境跑通。

---

## 6. 实现思路

### 6.1 总体策略：Python 黄金基准 → C++ 生产（双轨）

```
阶段 A：验证基准（Python）                 阶段 B：生产实现（C++，brpc + bthread）
┌──────────────────────┐           ┌──────────────────────────┐
│ Python 单机参照实现     │  数值对齐   │ C++ 生产                   │
│ ・模型/引擎/存储/并发    │ ────────▶ │ ・接入/编排/worker 服务       │
│ ・等价性断言（验证矩阵）  │  1e-4→1e-6 │ ・PS（独立部署）             │
│ ・存储无关 KVStore 协议  │           │ ・混合参数化 op（原生/vLLM）   │
└──────────────────────┘           └──────────────────────────┘
```

1. **先数值、后工程**：任何 C++ 移植（算子 → worker/服务 → vLLM op）都必须与 Python 基准 `max|diff|` 对齐，Python 实现是黄金基准（验证方法学见 model_design §6）；
2. **先正确、后性能**：语义正确性（等价性/一致性/降级）先于性能优化收口；
3. **契约先行**：存储后端、指标出口、PS wire 协议都以 Protocol/proto 先行固化，实现可替换。

### 6.2 两阶段等价性（算法前提）

等价性论证与三不变量 I1~I3（因果隔离/投影一致性/掩码重构一致）属模型层内容，**见 [model_design.md §4](./model_design.md)**。对工程的意义：等价性成立是「把 S 编码移出在线关键路径」这一拆分合法性的算法前提；任何破坏 I1~I3 的工程改动（掩码实现、投影复用、序列化裁剪）都直接破坏正确性，必须回归验证矩阵。

### 6.3 零拷贝数据面思路

| 环节 | 常规做法（拷贝） | 本实现（零拷贝） |
|---|---|---|
| 序列化写 | 逐层 `bytes` 拼接 O(n²) | 预分配单一 `bytearray` + `ctypes.memmove(data_ptr → buf+offset)` 单次直搬 |
| 反序列化读 | `bytearray(payload)` 副本 → tensor | `torch.frombuffer(payload, offset=pos)` 直接视图；`bytes` 只读 / `mmap·memoryview` 可写回 |
| 进程内存储 | dict 存 bytes，读时再拷 | mmap 模式：payload 落盘 + `mmap(ACCESS_WRITE)` 映射，record 暴露 memoryview，**读侧免二次拷贝** |
| 跨进程 | socket 逐字节搬运 | 生产由 datasystem 共享内存/HBM 直通承接（adapter 已留 `prefetch` 口） |

### 6.4 路由与本地化思路

同一 `user_id` 三处使用必须同桶：① KV 分片（`ShardedKVStore.shard_of`）② worker 派发（`WorkerPool.worker_for`）③ 扩缩容迁移评估（`remap_ratio`）。统一复用 `Router`（jump 哈希）且 `num_shards == num_workers`，保证「KV 与 owner worker 同节点共存」（D 决策：读命中本地内存而非跨节点 RDMA）。

---

## 7. 实现设计

### 7.1 软件架构（详细架构图）

#### 7.1.1 生产架构组件视图（C++/brpc，工程级目标形态 —— 主视图）

> 这是本系统的**主架构视图**：入口、编排、热路径全部为 C++（与主流生产精排一致）。
> Python 参照实现（§7.1.2）仅承担数值基准与协议验证职责，**不是**系统入口。

```
        上游排序服务（召回/粗排下游）             行为流（Kafka/MQ，按 user_id 哈希分区）
        brpc/gRPC 客户端                          分区消费者组
             │ OnlineRank.Score(uid, 候选特征)         │ Ingest(uid, ΔL 行为特征)
             ▼                                            ▼
╔══════════ 接入层：Online / Nearline brpc Server（独立服务进程）════════════╗
║  bthread-per-RPC：M:N 协程调度（无解释器锁，单机十万级并发长连接）          ║
║  过载保护：max_concurrency 方法级限流 → 快速拒绝 ELIMIT（背压给调用方）      ║
║  超时/取消：Controller.timeout_ms + done 回调链（全链路 deadline 传递）     ║
║  服务发现：注册中心（模型版本/分片拓扑 → 调用方路由表）                    ║
╚═════════╤══════════════════════════════════════════════════╤═════════════╝
          │ jump(uid) 一致性哈希 = KV 分片桶号（数据本地化前提）  │ 同一哈希域
          ▼                                                    ▼
╔════════ 编排层：Dispatcher + 每 worker 独立有界 mpsc 队列 ══════════════════╗
║  N 个 C++ worker 线程：内核态并行调度，无 GIL；可绑核 / NUMA 亲和            ║
║  req_seq 异步匹配 + 乱序完成：done(Controller) 点对点回调，无共享出队      ║
║  Dynamic batching：满批 or max_wait 超时出批（[ΣM,·,·] 整批单次前向）      ║
╚═════════╤═════════════════════════════════════╤═══════════════════════════╝
          │ Stage I：encode_s（B=1）             │ Stage II：score_ns_batch（B=ΣM）
          ▼                                       ▼
╔════════ 引擎层：C++ TwoStageRunner（ATen 算子 → 自定义 CUDA kernel）═════════╗
║  混合参数化 op：W_s 共享投影（S 段）+ W_ns_list 逐 token 投影（NS 段）      ║
║  算子内真并行：at::parallel_for（intra-op 线程池/OpenMP）；逐层循环在 C++ 内  ║
║  KV 内存布局：per-layer 连续张量 + pyramid@read 尾部裁剪（读侧带宽优化）    ║
╚═════════╤═══════════════════════════════════╤══════════════════════════════╝
          │ put / append(offset+CAS)            │ mget（shard 内聚合批量读）
          ▼                                       ▼
╔════════ 数据面（全部 C++ 客户端）═══════════════════════════════════════════╗
║  KVStore SDK → datasystem 集群：HBM/DRAM/SSD 多级缓存 + RDMA 零拷贝传输     ║
║  EmbeddingPSClient → PS 集群（独立部署：brpc 多表 + Knuth 分片）          ║
║  MetaStore SDK → Redis Cluster（KVPointer + TTL + model_version）           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**各层职责**：

| 层 | 组件 | 职责要点 |
|---|---|---|
| 接入层 | Online/Nearline brpc Server | RPC 入口、bthread 承载、方法级限流、超时/取消、服务注册 |
| 编排层 | Dispatcher + mpsc 队列 | 路由派发、乱序完成、背压、动态攒批（§7.4） |
| 引擎层 | TwoStageRunner | 两阶段张量计算、混合参数化 op、算子内并行 |
| 数据面 | PS / KV SDK / Meta SDK | 存储语义、序列化、分片、元数据（§4.3/§7.5/§7.6） |
| 存储后端 | datasystem / Redis / MQ | 外部系统，经 SDK 接入，对上层透明 |

**与主流工程级精排的对齐点**：入口为 C++ RPC 服务（非 Python 进程）、协程化 IO（bthread）与真实多线程 worker、C++ 数据面客户端（PS/KV 直连，无 Python 中转）、热路径无解释器参与。Python 在本系统的合理位置只有两处：离线训练/实验，以及数值黄金基准（§7.1.2）。

#### 7.1.2 验证基准组件视图（Python harness —— 黄金基准，非生产入口）

设计定位：算法/协议/契约的**验证载体**（组件名即模块路径），承担黄金基准与契约固化职责；因解释器锁限制，其多线程只用于协议验证、不承载生产性能语义（性能以 §7.1.1 生产视图为准）。各组件的落地状态见 [implementation_status.md](./implementation_status.md)。

```
                    调用方线程（M 个）                     Nearline 触发源（行为流消费者）
                          │ submit(uid, payload)                    │ ingest(s_emb, s_mask, uid)
                          ▼                                         ▼
╔═════════════════════ 编排层（serving/dispatcher.py + pipeline.py）══════════════════════╗
║  Dispatcher ───────────────────────────────────────────────────────────────────────── ║
║   ├─ _seq_lock ──▶ req_seq 单调分配                                                    ║
║   ├─ _inflight{req_seq → Future} ── 异步匹配表（乱序完成）                              ║
║   └─ _choose_worker(uid) ──hash──▶ pool.worker_for(uid) ═══ Router(jump) 同桶路由      ║
║  BatchScheduler（FIFO 攒批：Condition + deque；满批或 max_wait 超时出批）                ║
╚═══════════╤════════════════════════════════════════════════════════════════╤═══════════╝
            │ try_enqueue / enqueue（有界队列，满=背压）                      │ next_batch
            ▼                                                              ▼
╔════════ WorkerPool（N worker 线程 × 独立有界 queue.Queue）═══════════════════════════════╗
║  W0 ◀─q0─┐                                                                W1 ◀─q1─┐ ... WN-1      ║
║   worker 主循环：q.get() 阻塞 → handler(req) → _on_done → Dispatcher._complete          ║
║   handler = NearlineWorker.ingest | OnlineWorker.score | score_batch                     ║
╚═════════════════╤═════════════════════════════════════════════════════════╤═════════════╝
                  │ Stage I: encode_s                    Stage II: score_ns(_batch)
                  ▼                                                  ▼
╔═════════════ 引擎层（serving/two_stage.py：TwoStageRunner）══════════════════════════════╗
║  encode_s: for block in L:                                                            ║
║    RMSNorm → W_s 投影 → 缓存(K_s^l,V_s^l) → SDPA(_s_attn_mask) → final_proj → 残差       ║
║    → network_s FFN → 残差 → pyramid 尾部裁剪（token+mask 同步）                          ║
║  score_ns: for block in L:                                                             ║
║    读 (K_s^l,V_s^l) expand→M → W_ns_list[i] 逐 token 投影 → cat([K_s∥K_ns])              ║
║    → SDPA(_cross_attn_mask) → final_proj → 残差 → networks_ns_list[i] FFN → 残差          ║
║  score_ns_batch: 逐层 stack 各用户 K/V（宽度 dims[l] 恒定）+ arange 掩码 → 同上           ║
║  head: ns[:,-Ns:,:].mean(1)（或 CLS）→ linear → logits [M,T] / [B,T]                    ║
╚═════════════════╤═══════════════════════════════════════════════════════╤═══════════════╝
                  │ serialize(零拷贝+m元数据)               deserialize/frombuffer │ get/mget
                  ▼                                                            ▼
╔════════════ 数据面层（存储无关协议 + 多后端 adapter）══════════════════════════════════════╗
║  KVStore(Protocol) ◀── build_kv_store(KVConfig) 工厂                                    ║
║   ├── LocalKVStore：dict + threading.Lock；mmap 模式 payload 落盘 + ACCESS_WRITE          ║
║   │     共享映射（读侧零拷贝）；append=offset+CAS 校验+逐层 cat+重建                        ║
║   ├── YuanrongKVStore：kv().set/get；get 经 deserialize_with_meta 恢复元数据       ║
║   │     append 读-合并-写 + expect_checksum CAS fencing                          ║
║   └── ShardedKVStore：jump 哈希分片门面，mget/delete/prefetch 按 shard 聚合               ║
║  MetaStore：KVPointer{checksum,s_len,per_layer_len,ts,obj_key} + TTL 惰性过期 + 校验       ║
║  EmbeddingPSClient：Knuth 分片查表 + seed 兜底 ◀──▶ deploy/ps（C++ brpc 多表 PS）        ║
║  ServingMetrics：timing（perf_counter→ms 直方图）/ count / gauge + p50/p99 快照           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
                  │                                          │
                  ▼                                          ▼
╔════════ 模型层 ═════════╗              ╔════════ 存储后端层 ═════════╗
║ OneTrans backbone      ║              ║ 本地：dict/mmap 文件        ║
║  CoreOneTransBlock × L ║              ║ 集群：yuanrong datasystem  ║
║  MixedCausalSelfAttn   ║              ║     （HBM/DRAM/SSD 多级）   ║
║  MixedFFN              ║              ║ PS：brpc 分片嵌入表         ║
║ OneTransTokenizer      ║              ║ 元数据：Redis/本地 dict     ║
╚════════════════════════╝              ╚═════════════════════════════╝
```

**该视图的职责边界**（为什么 Python 只能是验证载体而非生产实现）：解释器锁（GIL）使纯 Python 计算段（逐层/逐 token 循环）上的多线程完全串行，torch 算子虽在执行期释放锁，但被 Python 循环切碎后单算子太薄、可重叠部分有限——无法支撑并发吞吐与时延 SLO（详细分析见 [implementation_status.md](./implementation_status.md)）。因此 Python 参照实现**只**交付三类结论：

1. **数值黄金基准**：两阶段 vs 单前向等价性、序列化 roundtrip、append CAS 语义、掩码重构不变量——作为 C++ 移植的验收标准（验证方法学见 model_design §6）；
2. **并发协议验证**：req_seq 乱序匹配、有界队列背压、同 user 哈希串行化、攒批窗口语义——语言无关协议在真实并发交错下无竞态；
3. **契约固化**：KVStore 协议、payload 内存布局、PS wire 协议（proto）、指标语义。

**性能结论（吞吐/QPS/时延 SLO/扩展性）一律以 C++ 生产实现（§7.1.1）为准**，Python harness 的性能数字只作回归对照，不得外推。

#### 7.1.3 分层职责与依赖方向

| 层 | 生产实现（目标形态） | 验证基准（Python） | 职责 | 只依赖 |
|---|---|---|---|---|
| 接入 | brpc Online/Nearline Server | — | RPC 入口、限流、超时/取消 | 编排 |
| 编排 | Dispatcher + mpsc 队列 | dispatcher.py / pipeline.py(BatchScheduler) | 并发调度、攒批、请求生命周期 | 引擎、数据面 |
| worker | C++ worker 线程 | pipeline.py(Nearline/Online) | 阶段编排、降级、埋点 | 引擎、数据面、指标 |
| 引擎 | C++ TwoStageRunner | two_stage.py | 张量计算（编码/打分） | 模型层 |
| 数据面 | KV/Meta C++ SDK；独立 PS | kv_store/serialize/local_adapter/datasystem_adapter/sharded/meta_store | 存储语义、序列化、分片、元数据 | torch（张量字节） |
| 横切 | 路由（Knuth/jump 哈希跨语言同构）/ 指标 / 权重 | router.py / metrics.py / weight_loader.py | 路由、指标、权重 | — |

**关键解耦**：worker 只依赖 `KVStore` 协议与 `TwoStageRunner`，不感知后端；后端经 `build_kv_store(KVConfig)` 注入；指标经 `ServingMetrics`（满足 `MetricsSink` 协议可替换）。

### 7.2 数据流设计（DFD）

#### DFD-0（顶层）

```
 ┌─────────┐  行为事件(ΔL)                                候选分数 [M,T]  ┌──────────┐
 │ E1 行为流 │ ────────────▶ ┌──────────────────┐ ──────────────────────▶ │ E3 上游   │
 └─────────┘                │   精排服务系统      │                          │ 排序服务  │
 ┌─────────┐  打分请求       │                  │  ┌──────────┐           └──────────┘
 │ E2 打分  │ ─────────────▶│                  │◀─│ E4 模型   │ checkpoint/表版本
 │ 请求方   │                └───────┬──────────┘  │ 发布流程  │
 └─────────┘                        │ 查表          └──────────┘
                                    ▼
                          ┌──────────────────┐
                          │ fabric ① 稀疏 PS  │ embedding 向量
                          └──────────────────┘
```

#### DFD-1（一层分解：P 过程 / D 存储 / 数据流标号）

```
E1 行为流 ──f1: {uid, ΔL 特征, ts}──▶ P1 S-tokenize+查表 ──f2: s_emb,s_mask──▶ P2 Stage I 编码
                                     ▲│                                            │
                            f3: id→向量││（查 D1）                    f4: UserKV{per_layer, s_len, plen}
                                     │└──f3──────── D1 Embedding PS ◀──共享───────┐│
                                     ▼                                             ▼│
E2 打分请求 ──f5: {uid, mv, M 候选特征}──▶ P3 攒批调度器                P4 序列化(put/append)
                                     │                                  │ f6: payload bytes
                                     │ f7: ScoreRequest                 ▼
                                     ▼                        D2 UserKV Store ◀──f6
                                  P5 Online worker ──f8: get/mget──▶ D2 ──f9: UserKVRecord──▶ P5
                                     │                                        ▲
                                     │ f10: pointer 校验                       │ f14: pointer
                                     ▼                                        │
                                  D3 MetaStore ◀──f14─────────────── P6 元数据维护 ◀─（put/append 后）
                                     │
                  P5 ──f11: UserKV, ns_emb──▶ P7 Stage II 交叉注意力 ──f12: logits [ΣM,T]──▶ P8 结果还原
                                                                                  │ f13: 按调用方分发
                                                                                  ▼
                                                                              E3 上游排序
```

| 数据流 | 内容 | 生产者 → 消费者 |
|---|---|---|
| f1 | 行为事件（uid、ΔL 行为特征、时间戳） | E1 → P1 |
| f2 | `s_emb [1,S0,D]`、`s_mask` | P1 → P2 |
| f3 | 特征 ID → embedding 向量 | D1 → P1/P3 |
| f4 | `UserKV{per_layer[(K,V)]×L, per_layer_len, s_len}` | P2 → P4 |
| f5 | 打分请求（uid、model_version、M 候选） | E2 → P3 |
| f6 | payload bytes（magic+header+raw，含元数据） | P4 → D2 |
| f8/f9 | get/mget 请求 / UserKVRecord | P5 ↔ D2 |
| f10/f14 | KVPointer（checksum/len/ts/obj_key） | D3 ↔ P5/P6 |
| f11 | UserKV + ns_emb | P5 → P7 |
| f12 | logits `[ΣM, T]` | P7 → P8 |
| f13 | 按调用方还原的分数 | P8 → E3 |

### 7.3 核心处理过程（流程图）

#### 7.3.1 Nearline 写路径（P1→P2→P4）

```
┌──────────────────┐
│ 行为事件(ΔL) 到达  │
└────────┬─────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│ NearlineWorker.ingest(s_emb, s_mask, uid, mv, seq_ts_last) │
│                                                            │
│ ① timing("nearline.encode_stage1")                         │
│    kv = runner.encode_s(s_emb, s_mask)                      │
│      for l, block in enumerate(L):                          │
│        per_layer_len[l] = smask.sum()   # 先记有效长度        │
│        h = RMSNorm(s)                                       │
│        q,k,v = W_s(h).reshape(B,S,3,H,d).unbind(2)         │
│        per_layer[l] = (k, v)             # 缓存 K/V          │
│        attn = SDPA(q,k,v, mask=_s_attn_mask(smask))         │
│        z = s + final_proj(attn)                             │
│        z = z + network_s(RMSNorm(z))                        │
│        s   = z 尾部 out_seq_num 个（pyramid 裁剪）            │
│        smask = smask 尾部同步裁剪                            │
│      return UserKV(per_layer, per_layer_len, s_len)         │
│                                                            │
│ ② timing("nearline.append_kv")                              │
│    payload = serialize(kv.per_layer,                       │
│              s_len=kv.s_len,          # 元数据固化            │
│              per_layer_len=kv.per_layer_len)                │
│    rec = UserKVRecord(key=KVKey(mv,uid), s_len, plen,       │
│                       dtype, payload, ts, created_at)       │
│    res = store.put(rec)   # 或 append(delta) 见 7.3.4       │
│                                                            │
│ ③ count("nearline.events_ingested"); gauge("kv.obj_cnt")    │
└────────────────────────────┬───────────────────────────────┘
                             ▼
                      PutResult{accepted, version, checksum}
```

#### 7.3.2 Online 读路径（单请求，含 miss 降级）

```
┌──────────────────────────────┐
│ score(uid, mv, ns_emb[M,Ns,D])│
└──────────────┬───────────────┘
               ▼
┌────────────────────────────────────────────────────┐
│ OnlineWorker.score                                  │
│ ① timing("online.kv_get"): rec = store.get(KVKey)  │
│    ├─ rec is None?                                  │
│    │    count("kv.miss")                            │
│    │    return zeros([M,T])     # miss 降级：合法形状、│
│    │                              不抛异常            │
│    │    （上层按全零判定不可用 → 冷启动兜底路径）        │
│    └─ else count("kv.hit")                          │
│ ② timing("online.encode_stage2"):                   │
│    kv = decode_record(rec, device)                   │
│      # deserialize(payload) → frombuffer 零拷贝视图    │
│      # → .to(device)                                 │
│    logits = runner.score_ns(kv, ns_emb)             │
│      for l, block in enumerate(L):                   │
│        k_s,v_s = kv.per_layer[l]; expand→[M,...]     │
│        s_mask = [zeros(S_l-valid) ∥ ones(valid)]     │
│          # 左 padding：后 valid 列有效                 │
│        q,k,v = W_ns_list[i](ns[:,i,:]) 逐 token 投影  │
│        K = cat([k_s ∥ k_ns])  # 读缓存 + 在线拼接      │
│        attn = SDPA(q,K,V, mask=_cross_attn_mask)     │
│        z = ns + final_proj(attn)                     │
│        z = z + networks_ns_list[i](norm(z))          │
│    head: ns[:,-Ns:,:].mean(1) → linear → [M,T]       │
│ ③ count("online.qps"); count("candidate_throughput")│
└────────────────────────────┬───────────────────────┘
                             ▼
                     logits [M, T]
```

#### 7.3.3 攒批路径（score_batch + miss/hit 混批）

```
C1 ─submit(r1)─┐                      ┌─submit(r2) C2          submit(r3) C3
               ▼                      ▼                          ▼
        BatchScheduler._queue（deque + Condition）
               │ next_batch()：首个请求唤醒 → deadline = now + max_wait
               │ while 队列 < max_batch 且未超时: cond.wait(remaining)
               ▼
        batch = [r1, r2, r3]（满批或超时，≥1 条，时延有界）
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│ OnlineWorker.score_batch(batch)                                 │
│ ① recs = store.mget([r.key for r in batch])    # 一次批量读     │
│ ② 逐条分流：                                                    │
│    for rec, req in zip(recs, batch):                            │
│      m = req.ns_emb.shape[0]（该请求候选数）                       │
│      rec is None → miss_positions += [flat..flat+m)（记展平位置） │
│      else         → kvs += [kv]*m；embs += [ns_emb]（展平候选）   │
│      flat += m                                                  │
│ ③ out = zeros([ΣM, T])                                          │
│    若有 hit：logits = score_ns_batch(kvs, cat(embs))             │
│      # 逐层 stack K/V（宽度 dims[l] 恒定）→ [B,S_l,H,d]          │
│      # valid[B] → arange >= S_l-valid 掩码（左 padding 等价）     │
│    按展平顺序回填：i ∈ miss_set → 保持全零；否则 out[i]=logits[hit_ptr++]
│ ④ gauge("online.batch_size", ΣM)                                 │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
                    logits [ΣM, T]（顺序与提交一致）
```

#### 7.3.4 增量 append（offset + CAS 双校验）

```
┌────────────────────────────────────────────────────┐
│ store.append(DeltaKV{key, offset=旧s_len,          │
│              delta_len=ΔL, tensors[ΔK/ΔV]×L,       │
│              expect_checksum=上次返回的 checksum})   │
└───────────────────────┬────────────────────────────┘
                        ▼
              ┌── rec = get(key) ──▶ None?
              │                      └─是─▶ AppendResult(missing)
              ▼ 否
   offset == rec.s_len ?
     │否─▶ AppendResult(offset_conflict)     # 位置校验
     │是
     ▼
   expect_checksum 非空 且 != rec.checksum ?
     │是─▶ AppendResult(cas_conflict)         # CAS fencing
     │否
     ▼
   per_layer = deserialize(rec.payload)
   merged[l] = cat([K_s^l, ΔK_s^l], dim=1)   # 逐层尾部拼接
   new_rec = rebuild(s_len + ΔL, plen + ΔL)  # 重算 checksum
   put(new_rec)
     │
     ▼
   AppendResult(accepted, new_s_len, new_checksum)
   # 冲突时上层策略：读最新 → 重算增量重试，或全量 put 幂等重建
```

### 7.4 线程与并发模型（重点）

**定位声明**：本节描述的是**语言无关的线程协作协议**（职责/唤醒/协作/乱序匹配）。7.4.1~7.4.6 以 Python 参照实现为载体说明协议语义；生产实现为 C++（brpc bthread + N 个真实 worker 线程，见 7.4.7 映射）。**注意**：Python 受 GIL 限制，其多线程仅验证协议正确性（乱序/背压/同 user 串行化），不提供并行性能——性能语义以 7.4.7 的生产映射为准。

代码位置：[dispatcher.py](file:///workspace/onetrans/serving/dispatcher.py)、[pipeline.py](file:///workspace/onetrans/serving/pipeline.py)（BatchScheduler）。

#### 7.4.1 线程清单与职责

单机参照实现中的线程（以一次服务装配为例）：

| # | 线程 | 数量 | 创建方 | 负责的工作 | 实现方式 | 唤醒方式 | 退出方式 |
|---|---|---|---|---|---|---|---|
| 1 | 客户端线程 | M（调用方） | 用户代码 | `Dispatcher.submit` 提交请求；随后 `Future.result()` 阻塞等待 | `threading.Thread`（或任意上层线程） | 被 `fut.set_result/set_exception` 内部 Condition 唤醒 | 请求返回即结束 |
| 2 | Worker 线程 | N（=num_workers，daemon） | `WorkerPool.start()` | 阻塞取本队列请求 → 执行 handler（编码/打分）→ 回调 `_on_done` 完成匹配 | 独立 `threading.Thread`，每线程独享一个 `queue.Queue(maxsize=cap)` | 阻塞在 `q.get()`（内部 not_empty Condition），`put` 时被唤醒 | 收到哨兵对象 `_STOP` 后 `return`，外层 `join()` |
| 3 | 攒批消费线程 | 1（服务装配） | 服务装配代码 | `BatchScheduler.next_batch` 攒批 → 调 `score_batch` | `threading.Thread` + `threading.Condition` | `submit()` 的 `cond.notify()` 唤醒；超时（max_wait 窗口）自然醒 | 服务停止约定（当前为 daemon 随进程） |
| 4 | 行为流消费线程 | 1（nearline 侧） | 服务装配代码 | 循环取行为事件 → `NearlineWorker.ingest` | `threading.Thread`（或外部 MQ 消费者） | 事件源（MQ/队列）唤醒 | 同上 |
| 5 | 后端线程 | 视后端 | — | LocalKVStore 内部无独立线程（调用方线程内加锁）；datasystem/PS 的网络 IO 由其后端线程池承接 | — | — | — |

> 说明：`LocalKVStore`/`LocalMetaStore`/`ShardedEmbeddingTable` 均为**被动对象**（无自有线程），其线程安全靠锁（见 7.4.6）。

#### 7.4.2 实现方式（并发原语与数据结构）

```python
# dispatcher.py —— 核心并发数据结构
class WorkerPool:
    _queues: list[queue.Queue]      # N 个独立有界队列（maxsize=queue_capacity）
    _workers: list[Thread]           # daemon 线程，target=_run
    router: Router                   # jump 一致性哈希（worker_for 与 KV 分片同桶）
    _on_done: Callable               # 完成回调（由 Dispatcher 注册为 _complete）

class Dispatcher:
    _seq: int                        # 单调递增请求号
    _seq_lock: threading.Lock        # 保护 _seq（O(1) 临界区）
    _inflight: dict[int, Future]     # req_seq → Future 异步匹配表
    _inflight_lock: threading.Lock   # 保护 _inflight（O(1) 临界区）
    _rr / _rr_lock                   # round_robin 模式的游标（hash 模式不用）

# pipeline.py —— 攒批
class BatchScheduler:
    _queue: deque[ScoreRequest]      # FIFO 批队列
    _cond: threading.Condition       # 生产/消费协同（notify + wait(timeout)）
```

**设计要点**：
1. **消全局锁**：每 worker 一个独立队列，请求提交互不阻塞（对比「全局队列+单锁」会把所有提交串行化，队头阻塞放大 p99）；
2. **按 user 串行化免费获得**：hash 路由使同一 user 的请求恒落同一 worker → 同一用户的 KV 读写天然有序，无需用户级锁；
3. **有界队列 = 背压**：`put_nowait` 失败即拒绝（`OverloadRejected`），把过载信号异步传给调用方，而非无界排队拖垮时延；
4. **乱序完成**：响应不需按提交顺序返回，`req_seq` 是唯一关联键——快速请求不被慢请求阻塞。

#### 7.4.3 唤醒方式汇总

| 阻塞点 | 阻塞线程 | 底层机制 | 唤醒触发 |
|---|---|---|---|
| `q.get()` | Worker 线程 | `queue.Queue` 内部 `not_empty` Condition | 提交方 `put/put_nowait` 入队成功 |
| `fut.result()` | 客户端线程 | `Future` 内部 Condition | Worker 线程（经 `_complete`）`set_result` / `set_exception`；或背压路径 `set_exception(OverloadRejected)` |
| `cond.wait()`（首条前） | 攒批消费线程 | `BatchScheduler._cond` | 生产线程 `submit()` → `notify()` |
| `cond.wait(remaining)`（攒批窗口内） | 攒批消费线程 | 同上 + 超时参数 | 新请求 notify 提前唤醒（继续攒）或 `remaining` 耗尽自然醒（超时出批） |
| `q.get()` 收到哨兵 | Worker 线程 | 同队列机制 | `stop()` 对每队列 `put(_STOP)` → worker `return` → `join()` |

> 攒批窗口语义：`next_batch` 先**无限期等待首条**（保吞吐），首条到达后开始 `max_wait_seconds` 倒计时（保时延上界），窗口内继续攒到 `max_batch_size` 即满批出批。

#### 7.4.4 协作方式（线程间如何配合）

- **请求传递（生产者→消费者）**：客户端线程 → `queue.Queue` → Worker 线程；数据所有权随入队移交，worker 取出后独占处理（期间无锁）。
- **响应传递（消费者→生产者）**：Worker 线程算完 → `_on_done(req, result, worker_id)` → `Dispatcher._complete` 查 `_inflight` 表 pop 出对应 Future → `set_result(Response)`（或 `set_exception`）→ 客户端线程从 `result()` 返回。**响应不排队、不经过共享队列**，直接点对点唤醒。
- **乱序匹配协议**：`Request.req_seq` 由 Dispatcher 分配并随身携带；`Response.req_seq` 原样回带；`_inflight` 是唯一匹配表（pop 即完成移交，天然防重复完成）。
- **背压协作**：队列满 → `try_enqueue` 返回 False → Dispatcher pop `_inflight` 并 `set_exception(OverloadRejected)`。调用方在 Future 上感知失败，自行决定重试/降级/打点——**提交方永不因过载被同步阻塞**（可选 `timeout` 限时阻塞模式）。
- **异常传播**：Worker 主循环 `try: result = handler(req) except Exception as e: result = e`——handler 的异常不杀线程，作为「结果」交给 `_complete` → `set_exception` → 调用方在 `result()` 处感知。**worker 线程对业务异常免疫**。
- **攒批协作**：多客户端线程并发 `submit`（cond 锁内 append + notify）→ 单消费线程 `next_batch` 攒批 → 批结果按展平顺序切分回各调用方（由上层编排层分发）。

#### 7.4.5 具体请求端到端走读（在线打分，user="u1001"，M=200 候选）

时序（左侧为执行线程；W2 = worker 2）：

```
客户端线程 C1                    Dispatcher                     Worker2（q2 空闲阻塞中）         KVStore(shard2)
   │                                │                                │  ◀── q2.get() 阻塞          │
   │── submit("u1001", payload) ───▶│                                │                             │
   │                                │ ① seq=_alloc_seq()=42          │                             │
   │                                │   [_seq_lock: +1, 纳秒级]       │                             │
   │                                │ ② _inflight[42]=fut             │                             │
   │                                │   [_inflight_lock: 1 次 dict 写]│                             │
   │                                │ ③ w=worker_for("u1001")        │                             │
   │                                │   =Router.route → jump 哈希 = 2 │                             │
   │                                │ ④ q2.put_nowait(req) ──────────▶│  ◀── not_empty 唤醒          │
   │◀── return fut ─────────────────│                                │                             │
   │   （C1 此后可继续提交别的请求，    │                                │ ⑤ req = q2.get()            │
   │     或阻塞在 fut.result()）      │                                │ ⑥ handler=OnlineWorker.score │
   │                                │                                │   timing(online.kv_get):     │
   │                                │                                │   get(KVKey(mv,"u1001")) ──▶│
   │                                │                                │   ◀── UserKVRecord ─────────│
   │                                │                                │   count(kv.hit)              │
   │                                │                                │ ⑦ timing(online.encode_stage2):
   │                                │                                │   kv=decode_record(rec)      │
   │                                │                                │     # frombuffer 零拷贝视图   │
   │                                │                                │   logits=score_ns(kv,ns_emb)│
   │                                │                                │     # [200,2]，L 层交叉注意力  │
   │                                │                                │ ⑧ _on_done(req,logits,2)     │
   │                                │◀─ _complete(req_seq=42,...) ────│   # worker 立即回到 ⑤ 取下一条│
   │                                │ ⑨ pop _inflight[42] → fut       │                             │
   │                                │    fut.set_result(Response)     │                             │
   │◀── fut.result() 唤醒 ───────────│                                 │                             │
   │    = Response(req_seq=42,       │                                │                             │
   │      user_id="u1001",           │                                │                             │
   │      result=logits[200,2],      │                                │                             │
   │      worker_id=2)               │                                │                             │
```

**并发交错示例**（证明乱序完成与背压）：C1 提交 u1001（seq=42→W2）后，C2 立即提交 u2002（seq=43→W0）。若 u2002 的 KV 更小、W0 更快，则 `Response(43)` 先于 `Response(42)` 完成——各 Future 独立，互不阻塞。若 W2 队列已满（cap=8，已积压 8 条），C1 的 `put_nowait` 失败 → `_inflight.pop(42)` → `fut.set_exception(OverloadRejected(42))` → C1 在 `result()` 处捕获，选择重试或降级，同时可打点 `overload` 指标。

**攒批变体走读**：C1/C2/C3 分别 `BatchScheduler.submit`（各自 cond 锁内 append+notify）；消费线程被首条唤醒后开 5ms 窗口，窗口内凑齐 r1,r2,r3 → `score_batch` 一次 `mget(3 keys)`（shard 内聚合）+ 一次 `score_ns_batch` 前向（[ΣM,Ns,D]）→ `[ΣM,T]` → 编排层按各请求的 M 切分分发回各 Future。

#### 7.4.6 锁与临界区分析

| 锁 | 所属 | 保护对象 | 临界区 | 争用度 |
|---|---|---|---|---|
| `_seq_lock` | Dispatcher | `_seq` 自增 | O(1)，纳秒 | 低（提交频率级） |
| `_inflight_lock` | Dispatcher | `_inflight` dict 写/删 | O(1)，纳秒 | 低（每请求 2 次：注册+完成） |
| `_rr_lock` | Dispatcher | round_robin 游标 | O(1) | 仅 rr 模式 |
| 队列内部锁 ×N | WorkerPool | 各队列 ops | O(1)，纳秒 | 分散到 N 个队列，天然分片 |
| `LocalKVStore._lock` | local 后端 | dict 写路径（put/append/delete） | put：持久化全程；append：反序列化+拼接+重序列化（微秒~毫秒） | 中（按 user 路由后已分散；生产由 datasystem 后端并发承接） |
| `_locks[s]` ×S | ShardedEmbeddingTable | 单分片表读写 | O(1)，微秒 | 低（64 分片摊薄） |
| `_cond` | BatchScheduler | deque 批队列 | O(批大小) | 低（攒批节奏） |
| `LocalMetaStore._lock` | 元数据面 | 指针 dict | O(1) | 低 |

**无锁路径**：worker 取到请求后的全部计算（`encode_s`/`score_ns`/张量操作）**无共享可变状态**（权重只读、UserKV 每请求独立对象）——计算时间是零锁占用的；`ServingMetrics` 依赖 CPython GIL 下 `defaultdict`/`list.append` 的原子性（轻量实现的已知取舍，生产替换为分桶无锁聚合）。

#### 7.4.7 生产形态映射（brpc + bthread）

| Python 参照 | 生产 C++（brpc + bthread） | 语义对应 |
|---|---|---|
| `Dispatcher.submit` | RPC 入口 handler（bthread 承载，M:N 调度） | 每请求一个 bthread，提交即返回 Controller/done |
| `WorkerPool` N 队列 | `ExecutionQueue`/每 worker 有界队列 + bthread 工作组 | 独立队列消全局锁 |
| `Future` + `_inflight[req_seq]` | brpc `Controller` + `done` 回调 / `bthread_id` | 异步匹配、乱序完成、超时（timeout_ms） |
| `try_enqueue` Full → `OverloadRejected` | `max_concurrency` 限流 + 快速拒绝（ELOGOFF/ELIMIT） | 背压 |
| `_STOP` 哨兵 + `join` | `Server.Stop(...)`（先停收新请求，再 drain 队列） | 优雅停机（drain & wait 语义） |
| `LocalKVStore._lock` | datasystem 后端并发（分片锁/无锁结构） | 存储层承接 |
| `BatchScheduler` | 服务端 dynamic batching（或 brpc 批量接口 + 合并器） | 满批或超时 |

**与 LLM 推理服务的线程模型差异**：Stage II 非自回归（M 候选整批并行、单次前向出全部分数），无逐 token 循环与 continuous batching 的调度复杂度；并发重心是「高并发读 KV + 交叉注意力吞吐 + 数据本地性」，而非 decode 的 token 级流水。

### 7.5 关键数据结构

```python
# ---- 键与结果（kv_store.py）------------------------------------------- #
@dataclass(frozen=True)
class KVKey:                    # 逻辑主键（不可变，可哈希）
    model_version: str
    user_id: str
    # __str__ → "kv:{b64(mv)}:{b64(uid)}"（datasystem 字符集规范）

@dataclass
class UserKVRecord:             # 跨存储边界的完整对象（序列化形态）
    key: KVKey
    s_len: int                  # 有效历史长度（append 的 offset 基准）
    per_layer_len: list[int]    # 每层有效 token 数（左 padding 掩码的唯一依据，R-1）
    dtype: str                  # "float16" | "bfloat16" | "float32"
    payload: bytes              # §7.7 布局的字节 blob
    seq_ts_last: int = 0        # 最近行为时间戳
    created_at: int = 0
    @property
    def checksum(self) -> str:  # sha256(payload)，内容指纹（CAS fencing token）

@dataclass
class DeltaKV:                  # 增量 append 载荷
    key: KVKey
    base_version: str
    offset: int                 # 必须等于当前 rec.s_len（乐观校验）
    delta_len: int              # ΔL
    tensors: list[tuple[Tensor, Tensor]]     # 每层 (ΔK_s^l [1,ΔL,H,d], ΔV_s^l)
    expect_checksum: str = ""   # 非空时要求 == 当前 payload checksum（CAS fencing，R-2）

@dataclass
class PutResult:    accepted: bool; version: str; checksum: str; reason: str = ""
@dataclass
class AppendResult: accepted: bool; new_s_len: int; checksum: str; reason: str = ""
                   # reason ∈ {ok, missing, offset_conflict, cas_conflict, layer_mismatch}

# ---- 元数据指针（meta_store.py）--------------------------------------- #
@dataclass
class KVPointer:               # fabric ③：小对象、高频读
    model_version: str; user_id: str
    checksum: str; s_len: int; per_layer_len: list[int]
    seq_ts_last: int = 0; obj_key: str = ""; created_at: int = 0
# validate_pointer(rec, ptr): checksum/s_len/per_layer_len 三者一致才 True

# ---- 引擎态（two_stage.py）-------------------------------------------- #
@dataclass
class UserKV:                   # Stage I 产出的内存形态（B=1 单用户）
    per_layer: list[tuple[Tensor, Tensor]]   # (K_s^l [1,S_l,H,d], V_s^l) × L
    per_layer_len: list[int]
    s_len: int

# ---- 并发原语（dispatcher.py）----------------------------------------- #
@dataclass
class Request:  user_id: str; payload: Any; req_seq: int = -1
@dataclass
class Response: req_seq: int; user_id: str; result: Any; worker_id: int = -1
# OverloadRejected(req_seq)：背压信号（Future 异常承载）

# ---- 攒批（pipeline.py）----------------------------------------------- #
@dataclass
class ScoreRequest:            # 一次打分请求（一个 user 的 M 个候选）
    key: KVKey
    ns_emb: Tensor             # [M, Ns, D]

# ---- C++ PS 侧（deploy/ps/embedding_server.cc，概念对应）--------------- #
# ShardedEmbeddingTable: num_shards × (mutex + unordered_map<int64, vector<float>>)
# TableRegistry: table_name → ShardedEmbeddingTable（懒建 + 淘汰），每表独立 version_
```

### 7.6 核心伪代码

#### 7.6.1 Worker 主循环（`WorkerPool._run`）

```python
def worker_loop(worker_id, q, handler):
    while True:
        req = q.get()                    # 阻塞点①：not_empty Condition（提交方 put 唤醒）
        if req is STOP_SENTINEL:         # stop() 投递的哨兵
            return                       # 线程自然退出（外层 join）
        try:
            result = handler(req)        # 计算阶段：无锁独占（Nearline.ingest/Online.score）
        except Exception as e:
            result = e                   # 异常不杀线程，作为结果传播
        on_done(req, result, worker_id)  # → Dispatcher._complete：点对点唤醒调用方
```

#### 7.6.2 提交与完成（`Dispatcher.submit` / `_complete`）

```python
def submit(user_id, payload, timeout=None):
    seq = alloc_seq()                            # 临界区①：_seq_lock（纳秒）
    fut = Future()
    with inflight_lock:                          # 临界区②：注册匹配表
        inflight[seq] = fut
    req = Request(user_id, payload, seq)
    w = pool.worker_for(user_id)                 # jump 哈希（纯计算，与 KV 分片同桶）
    ok = pool.enqueue(w, req, t) if 有限时 else pool.try_enqueue(w, req)
    if not ok:                                   # 队列满 → 背压
        with inflight_lock:
            inflight.pop(seq)
        fut.set_exception(OverloadRejected(seq)) # 异步失败信号（不抛同步异常）
    return fut                                   # 调用方：fut.result() 阻塞或回调

def _complete(req, result, worker_id):           # worker 线程上下文执行
    with inflight_lock:                          # 临界区③：pop 即完成移交（防重复完成）
        fut = inflight.pop(req.req_seq, None)
    if fut is None or fut.done():
        return                                   # 已超时/已拒绝：丢弃即可
    if isinstance(result, Exception):
        fut.set_exception(result)                # 唤醒点：调用方 Condition
    else:
        fut.set_result(Response(req.req_seq, req.user_id, result, worker_id))
```

#### 7.6.3 攒批窗口（`BatchScheduler.next_batch`）

```python
def next_batch():
    with cond:
        while not queue:                  # 阶段①：无限期等首条（保吞吐）
            cond.wait()                   #   唤醒：submit() notify
        deadline = now() + max_wait       # 阶段②：开时延窗口（保上界）
        while len(queue) < max_batch:
            remaining = deadline - now()
            if remaining <= 0: break
            cond.wait(timeout=remaining)  #   提前醒（新请求）→ 继续攒；超时醒 → 出批
        return [queue.popleft() for _ in range(min(max_batch, len(queue)))]
```

#### 7.6.4 append CAS（后端语义，local/datasystem 一致）

```python
def append(delta):
    with lock:                                   # local 后端的存储锁
        rec = get(delta.key)
        if rec is None:            return AppendResult(missing)
        if delta.offset != rec.s_len:            return AppendResult(offset_conflict)
        if delta.expect_checksum and delta.expect_checksum != rec.checksum:
                                                return AppendResult(cas_conflict)  # R-2
        per_layer = deserialize(rec.payload)
        if len(per_layer) != len(delta.tensors): return AppendResult(layer_mismatch)
        merged = [(cat(K, dK, dim=1), cat(V, dV, dim=1))           # 逐层尾部拼接
                  for (K,V),(dK,dV) in zip(per_layer, delta.tensors)]
        new_rec = rebuild(s_len=rec.s_len + delta.delta_len,       # 重算 checksum
                          per_layer_len=[pl + delta.delta_len for pl in rec.per_layer_len])
        persist(new_rec)
        return AppendResult(ok, new_rec.s_len, new_rec.checksum)
```

#### 7.6.5 两阶段引擎（浓缩）

```python
def encode_s(s_emb, s_mask):                    # Stage I（B=1）
    s, smask, per_layer, plen = s_emb, s_mask, [], []
    s_len = smask.sum()
    for block in blocks:                         # L 层
        plen.append(smask.sum())                 # 该层输入的有效长度（先记）
        h = RMSNorm(s)
        q, k, v = W_s(h).reshape(B,S,3,H,d).unbind(2)
        per_layer.append((k, v))                 # ★ 缓存点
        z = s + final_proj(SDPA(q,k,v, mask=s_attn_mask(smask)))
        z = z + network_s(RMSNorm(z))
        s = z[:, -out_seq_num:, :]               # pyramid 尾部裁剪（token+mask 同步）
        smask = smask[:, -out_seq_num:, :]
    return UserKV(per_layer, plen, s_len)

def score_ns(kv, ns_emb):                        # Stage II（B=M 候选）
    ns = ns_emb
    for l, block in enumerate(blocks):
        k_s, v_s = kv.per_layer[l]; k_s = k_s.expand(M,...); v_s = v_s.expand(M,...)
        s_mask = cat([zeros(S_l - valid_l), ones(valid_l)])       # 左 padding 重构
        q,k,v = 逐 token W_ns_list[i](ns[:,i,:]) → cat           # NS 独立投影
        K, V = cat([k_s, k], dim=1), cat([v_s, v], dim=1)          # 读缓存 + 在线拼接
        z = ns + final_proj(SDPA(q,K,V, mask=cross_attn_mask(s_mask, Ns)))
        z = z + 逐 token networks_ns_list[i](RMSNorm(z))
        ns = z
    return linear(ns[:, -Ns:, :].mean(1))        # [M, T]
```

### 7.7 payload 内存布局与序列化

```
payload = <magic 12B> <header_len 4B(LE u32)> <header_json> <raw_bytes>
  magic       = b"ONETRANSKV\x01"
  header_json = {
    "dtype": "float16", "n_layers": L,
    "s_len": 31,                                  ← 有效长度固化（旧数据缺省→满宽）
    "layers": [
      {"l":0, "k_shape":[1,50,4,32], "v_shape":[1,50,4,32], "len":31},   ← 每层有效 len
      {"l":1, "k_shape":[1,28,4,32], "v_shape":[1,28,4,32], "len":27},
      ...
    ]
  }
raw_bytes = concat(K_s^0, V_s^0, K_s^1, V_s^1, ..., K_s^{L-1}, V_s^{L-1})   # 均 bshd 连续
每层字节数 = 2 × H × S_l × d × itemsize(dtype)
```

- **写侧零拷贝**：预分配总长 `bytearray`；`ctypes.memmove(buf+offset, t.data_ptr(), n)` 逐张量单次直搬（无中间 bytes 对象、无 O(n²) 拼接）。
- **读侧零拷贝**：`torch.frombuffer(payload, dtype, count, offset=pos).reshape(shape)` 直接视图；`bytes` 只读、`bytearray/memoryview/mmap` 可写回（与底层共享存储）。
- **元数据 API**：`read_header`（只解析头部，供 datasystem get/append 免整对象反序列化）、`deserialize_with_meta`（返回 `(per_layer, s_len, per_layer_len)`，旧数据回退满宽）、`per_layer_offsets`（层偏移，供 pyramid@read 部分读取）。
- **容量量级**（默认配置 fp16、linear dims 100→10、L=6）：`C_kv ≈ 2·H·d·2B · Σdims[l] ≈ 1024·330 ≈ 338 KB/用户`。

### 7.8 可靠性与可观测性设计

| 能力域 | 设计机制 |
|---|---|
| 降级 | KV miss → 全零 + `kv.miss` 打点（单/批一致）；权重缺失/损坏 → seed 兜底；PS miss → 确定性 seed 嵌入；上层按全零判定不可用走冷启动兜底 |
| 背压 | 双层：接入层 `max_concurrency` 方法级限流（快速拒绝）+ 编排层有界队列满 → `OverloadRejected`（异步失败信号，提交方永不被同步阻塞） |
| 重试与幂等 | 读操作（get/mget）幂等可安全重试；写操作（put 幂等覆盖；append 非幂等靠 offset+CAS 拒绝重放） |
| 一致性 | append offset+CAS fencing；pointer checksum/s_len/per_layer_len 三者一致才校验通过；序列化魔数校验；后端具备原生原子 CAS 时直通 |
| 版本化 | model_version 进 KV key；PS 每表独立 version；checkpoint 按 mv 命名；注册中心维护版本→实例路由 |
| 超时/取消 | 全链路 deadline 传递（Controller.timeout_ms / Future timeout）；过时结果在 `_complete` 处丢弃（pop 后 `fut.done()` 判定） |
| 熔断/健康 | 按后端（datasystem/PS）错误率熔断 + `/healthz` 依赖探针 |
| 优雅停机 | 先停收新请求 → 排空队列（drain & wait）→ join；哨兵对象触发 worker 自然退出 |
| 指标 | `nearline.encode_stage1` / `nearline.append_kv` / `online.kv_get` / `online.kv_mget` / `online.encode_stage2` / `kv.hit|miss` / `kv.obj_cnt` / `online.qps` / `online.candidate_throughput` / `online.batch_size`；导出到 Prometheus/OTel（分桶直方图） |
| 日志/追踪 | req_id/trace_id/user_id 贯穿 Nearline→Online→PS 的结构化日志与分布式 trace |

### 7.9 部署视图

```
                  行为流（按 user 分区）                  打分请求（按 user 路由）
                        │                                     │
 ┌────── Nearline Pool（C++ brpc 服务）──────┐   ┌────── Online Pool（C++ brpc 服务）──────┐
 │  Stage I：tokenize → S 编码 → put/append  │   │  Stage II：读 KV → NS 编码 → 交叉打分   │
 └──────────────┬───────────────────────────┘   └──────────────┬─────────────────────────┘
                │ 写（fabric ②）                                │ 读（同 shard，本地命中）
                ▼                                               ▼
 ┌────────────── UserKV datasystem（按 user 一致性哈希分片；HBM/DRAM/SSD 多级）─────────────┐
 └──────────────────────────────────┬──────────────────────────────────────────────────────┘
                                    │ 稀疏查表（fabric ①，S/NS 共享）
                                    ▼
 ┌────── 独立稀疏 PS（C++ brpc：Knuth 分片 × 多表 × 版本）──────┐   ┌── fabric ③ 元数据/版本 ──┐
 └─────────────────────────────────────────────────────────────┘   │ KVPointer / TTL / 注册  │
                                                                    └─────────────────────────┘
```

扩缩容：jump 哈希 remap O(k/n)（仅迁移 k/n 比例对象）；`num_shards == num_workers` 维持 KV/worker 同桶共址。

**为什么 Nearline/Online 分离部署**：
- **负载形态不同**：近线「写密集、批量尾巴」，在线「读密集、低时延、高 QPS」，拆开可独立扩容/限流/灰度；
- **稀疏参数独立服务化**：稀疏表可超单机内存，每实例各持一份成倍浪费；独立 PS 供两池共享查表；
- **UserKV 与 owner 同节点共存**：一致性哈希把 user 路由到 owner，读命中本地缓存，最大化数据本地性。

---

## 8. 与现状文档的边界

本文到此为止均为**目标设计**。各组件/各层的落地状态、实测结果、差距分级与推进路线**一律**见：

- [implementation_status.md](./implementation_status.md)——已实现与验证内容、分层落地状态、验证结果摘录；
- [gap_analysis.md](./gap_analysis.md)——差距清单（编号 G1~G14）、必要性分析与里程碑路线图。

---

## 9. 文档三分体系对应表

| 类别 | 文档 | 与本文关系 |
|---|---|---|
| ① 模型层 | [model_design.md](./model_design.md) | 本文 §4.2 的算法依据（等价性 I1~I3、掩码、pyramid 约束） |
| ② 端到端设计（概要） | [e2e_design_spec.md](./e2e_design_spec.md) | 本文的上游边界与决策 D1~D6 |
| ② 端到端设计（详细） | **本文** | 全量系统设计 |
| ③ 现状 & 差距 | [implementation_status.md](./implementation_status.md) | 本文各组件的落地状态与实测 |
| ③ 现状 & 差距 | [gap_analysis.md](./gap_analysis.md) | 差距分级、必要性分析与路线图 |
