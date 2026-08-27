# 序列 Transformer 精排（OneTrans 类）端到端设计说明书

> 版本：v0.1（概要设计）
> 文档类别：**② 端到端设计（概要）**——只依据设计意图给出系统边界、数据 fabric 与设计决策 D1~D6；详细设计见 [detailed_design.md](./detailed_design.md)，模型层见 [model_design.md](./model_design.md)，现状与差距见 [implementation_status.md](./implementation_status.md) / [gap_analysis.md](./gap_analysis.md)。
> 目标：以工业集群为研究对象，刻画「序列 Transformer 精排」的负载特征、端到端时延、硬件资源利用率与性能瓶颈。

---

## 1. 引言

### 1.1 目的

本说明书给出一个**集群级序列 Transformer 精排系统**的端到端设计骨架。研究对象是 OneTrans 这一类的
「用户侧摊销序列编码 + 候选条件化交叉注意力」排序算法，目标不是训练出最优模型，而是：

1. 复现算法的**两阶段（S 侧编码 / NS 侧交叉打分）**与 **KV Cache** 语义；
2. 刻画其**负载特征**（稀疏 gather、S 编码算力、KV 容量/带宽、候选交叉注意力算力）；
3. 量化**端到端时延、硬件资源利用率、性能瓶颈**，并与 DeepFM/DCNv2 等成熟精排做**等约束公平对比**。

### 1.2 范围

- **在范围内**：Nearline（Stage I）、Online（Stage II）、分布式 KV 存储、稀疏 Embedding 数据面、端到端数据流、指标采集。
- **不在范围内**：召回/粗排、在线 A/B 平台、完整训练参数服务器与异步训练、广告计费/控制面。

### 1.3 术语

| 术语 | 含义 |
|---|---|
| S-token / NS-token | 序列 token（用户行为历史）/ 非序列 token（用户/候选/上下文特征） |
| Nearline / Stage I | 用户历史编码，产出逐层 K/V（对应 LLM prefill） |
| Online / Stage II | 候选打分，读 K/V 做交叉注意力（对应 LLM decode，但并行非自回归） |
| User KV | 每用户、每层的 `(K_s^l, V_s^l)` 张量 + 元数据 |
| ΔL | 用户上一次编码后新增的行为条数（增量 append） |
| pyramid@read | 金字塔剪尾只作为 Online 读侧优化，近线缓存全序列 K/V |
| fabric | 一张独立的分布式「数据平面」，见 §4 |
| 单卡=单任务 | 单个推理任务模型规模小（单 GPU），但系统整体是集群 |

---

## 2. 目标与非功能需求

### 2.1 研究目标（业务无关的度量目标）

- **G1**：建立「三瓶颈 → 三组件」的可测量映射（稀疏 gather / S 编码 / KV+交叉注意力）。
- **G2**：量化 KV 两阶段拆分的**摊销收益**（容量、带宽、时延）。
- **G3**：在**等特征、等稀疏表、等约束**下与 DeepFM/DCNv2 做 **Pareto 前沿**对比。

### 2.2 非功能需求

| 维度 | 要求 |
|---|---|
| 时延 | 打分的 p50/p99/p999 可测、可分解到阶段 |
| 吞吐 | 给定 SLO 下的 QPS，及 QPS 扫描 |
| 资源 | GPU SM 利用率、HBM 带宽/容量、DRAM/SSD、RDMA 网络 |
| 可复现 | 固定硬件规格、seed、依赖版本、确定性 workload 生成器 |
| 可扩展 | 单卡 → 单机多卡 → 多节点，逐步 scale-out |
| 可观测 | 每个瓶颈维度有明确埋点（见详细设计 §8） |

---

## 3. 系统总体架构

```
                        行为流(按 user 分区)                请求(按 user 路由)
                             │                                     │
  ┌──────── Nearline Pool ─────────┐                ┌──────── Online Pool ─────────┐
  │ GPU0  GPU1  ...  GPUn           │                │ GPU0  GPU1  ...  GPUm         │
  │ Stage I：tokenize + S 侧编码    │                │ Stage II：NS 编码 + 交叉打分    │
  └──────────────┬─────────────────┘                └──────────────┬────────────────┘
                 │ 写 K/V (put/delta)                               │ 读 K/V (get)
                 ▼                                                  ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                    分布式 KV 存储（datasystem，按 user 分片）                        │
   │   对象(KV)语义 + HBM/DRAM/SSD 多级缓存 + 副本 + 跨节点零拷贝/传输                     │
   └────────────────────────────────────────────────────────────────────────────────┘
```

**核心抽象**：KV 存储被抽象为一个 **datasystem**（见详细设计 §2），内部是
「内存为中心、近计算、多级缓存、对象（KV）/流」语义，底层存储后端对上层透明。

---

## 4. 三大数据 fabric 与职责

集群化后，系统里存在三张**相互独立**的分布式数据平面，必须拆清，避免混成一个存储系统：

| fabric | 键 | 读/写画像 | 承载 | 是否参数服务器 |
|---|---|---|---|---|
| ① 稀疏 Embedding fabric | 特征值 ID | 读密集、超大、偏斜 | TorchRec 分布式 ShardedEmbedding | 是（PS 读侧） |
| ② 用户 KV fabric | `(model_version, user_id)` | nearline 写 + online 读、append、按 user 分片 | **datasystem（分布式 KV 存储）** | 否，但设计手法同构 |
| ③ 元数据/版本 fabric | `user_id` | 小对象、高频读、TTL/失效 | Redis Cluster（或 datasystem 元数据能力） | 否 |

- ①②③ 各自独立扩容；② 是本研究主战场，① 是上游数据、③ 是控制面。
- 「特征交互」由 backbone（transformer）承担；「特征表示」由 ① 承载。

---

## 5. 关键设计决策（ADR 摘要）

| 编号 | 决策 | 理由 |
|---|---|---|
| D1 | **单卡=单任务模型规模，系统=集群** | 模型小（单 GPU 可跑）；但 KV 必须分布式 |
| D2 | **pyramid@read**：近线缓存全序列每层 K/V，不剪尾 | 增量 append 干净正确，剪尾只作读侧带宽优化 |
| D3 | **KV 存储用 datasystem**：对象(KV)/流语义、多级缓存、副本、跨节点传输 | 复用成熟分布式存储底座 |
| D4 | **三 fabric 分离**（见 §4） | 避免「稀疏表 / 用户 KV / 元数据」耦合 |
| D5 | **推理侧只用分片 embedding gather，不建完整训练 PS** | 负载研究聚焦推理；训练 PS 属后续扩展 |
| D6 | **算法/工程策略映射 LLM 原生能力**（见详细设计 §5/§6） | 复用 vLLM 的 KV/PD/量化/调度，仅自研混合参数化层 |

---

## 6. 组件清单与边界

| 组件 | 职责 | 关键实现 | 边界 |
|---|---|---|---|
| Nearline Worker | 行为流增量 → S token → S 侧编码 → 写逐层 K/V | vLLM 自定义层（Stage I）+ TorchRec | 输出 = User KV 写请求 |
| Online Worker | 请求 → 读 User KV + 候选特征 → NS 编码 → 交叉注意力 → M 分数 | vLLM 自定义层（Stage II）+ TorchRec | 输出 = 候选分数 |
| KV 存储（datasystem） | User KV 的分布式持久/多级缓存/副本/传输 | yuanrong datasystem（或等价 adapter） | 见详细设计 §2 |
| Embedding 数据面 | 稀疏 ID → 向量，分片、频率感知 | TorchRec ShardedEmbedding | 被 Nearline/Online 共享 |
| 元数据/版本 | user → (version, len, ts, checksum, 冷层指针) | Redis Cluster | 控制面 |
| 编排/路由 | 请求按 user 路由、动态 batching、池调度 | Triton（或 vLLM 原生调度） | 不做 PD 语义 |

---

## 7. 端到端数据流

### 7.1 写路径（Nearline）

```
行为事件(ΔL) → 路由到 owner nearline worker
  → 读该 user 历史 + 增量 tokenize(S)
  → Stage I 编码（复用旧 K/V，仅算 ΔL 位置）
  → datasystem.append(user, version, ΔK/V)   [写 User KV]
  → 更新元数据(seq_ts_last, per_layer_len)
```

### 7.2 读路径（Online）

```
请求(uid) 到达 → 按 uid 路由到 online worker
  → datasystem.get(uid, version) 得到 User KV（或 ObjRef，零拷贝）
  → 特征服务取 M 个候选 NS 特征 + Embedding 查表(①)
  → Stage II：NS tokenize → 逐层交叉注意力(读 S K/V + 候选自身) → head
  → 返回 M 个 pCTR/pCVR
```

---

## 8. 技术选型（初版）

| 层 | 选型 | 备注 |
|---|---|---|
| 稀疏 Embedding | TorchRec `ShardedEmbedding`（jagged） | 频率感知分片 |
| Backbone/推理引擎 | vLLM（自定义 mixed-attention 层） | 复用 paged KV / prefix caching / 量化 |
| KV 存储 | yuanrong datasystem（对象/流、多级缓存、副本、跨机传输） | 或等价 adapter（Redis/S3/本地） |
| 元数据 | Redis Cluster | 或复用 datasystem 元数据 |
| 编排 | Triton Inference Server | 路由 + dynamic batching |

---

## 9. 风险与开放问题

| 风险 | 说明 | 缓解 |
|---|---|---|
| 混合参数化层在 vLLM 的落地 | 需自定义 attention/FFN，无法原生表达 NS token-specific 参数 | 先做单卡参照实现（数值基准），再移植 |
| datasystem 对「逐层张量 + append + 版本化」的适配 | 通用 KV 语义 vs 张量追加语义 | 定义薄语义层（详细设计 §2） |
| 金字塔@读 vs 全量读的带宽收益 | 未量化 | 作为首个消融实验 |
| KV 容量/带宽是否为主瓶颈 | 量级估算 40TB/800GB/s 级别 | 参数化容量/带宽模型先行 |
| 基线公平性 | DeepFM/DCNv2 需同 seam 实现 | 同 Embedding 路径、同约束 |

---

## 10. 交付物对应用

- 本说明书 → 概要边界与决策（D1~D6）。
- [详细设计文档](./detailed_design.md) → 接口契约、张量契约、指标采集点、逐组件设计。