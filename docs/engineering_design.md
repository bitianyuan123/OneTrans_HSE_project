# 序列 Transformer 精排系统 · 工程级详细设计

> 版本：v0.1
> 上游：[端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6）、[详细设计](./detailed_design.md)（KV/Tensor/指标契约）
> 本文承接前两份文档，聚焦**工程侧**的高并发、分布式落地方案：线程模型、独立参数服务器、两阶段 brpc 分离部署、KV 零拷贝数据面、一致性哈希路由/元数据失效、动态 batching。

---

## 0. 文档关系与定位

| 文档 | 关注点 |
|---|---|
| e2e_design_spec.md | 业务边界、数据 fabric、设计决策（D1~D6） |
| detailed_design.md | KV 存储接口契约、Tensor 契约、算法→工程映射、指标埋点 |
| **engineering_design.md（本文）** | **软件架构 + 并发 + 部署 + 数据面优化**（生产落地方案） |

本文不重复前面的接口/张量契约，只在需要处引用；新增的是「怎么在高并发、分布式环境下把它跑起来并可持续演进」的方案。

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

**核心结论**：Python 侧只承担「模型结构/权重加载/数值基准」三类职责；所有在线/近线热路径与稀疏参数服务，生产上由 C++（brpc + bthread M:N）承载。Python 实现是 C++ 移植的**黄金数值基准**——`demo.py` 的等价性断言（max|diff| 逐位收敛到 1e-6 量级）是 C++ 移植的验收门槛。

---

## 2. 两阶段 brpc 分离部署

### 2.1 部署拓扑

```
                        行为流(按 user 分区)               请求(按 user 路由)
                             │                                    │
  ┌──────── Nearline Pool（C++ brpc 服务）────────┐   ┌──────── Online Pool（C++ brpc 服务）────────┐
  │  Stage I：tokenize + S 侧逐层编码 → 写 UserKV │   │  Stage II：读 UserKV + NS 编码 + 交叉打分    │
  └───────────────┬───────────────────────────────┘   └───────────────┬──────────────────────────────┘
                  │ put/append                            │ get/mget
                  ▼                                       ▼
   ┌───────────────────────────  UserKV datasystem（fabric ②，按 user 分片）──────────────────────────┐
   └──────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                          │ 稀疏特征查表（S/NS 共享）
                                          ▼
   ┌─────────────── 独立稀疏参数服务器 PS（C++ brpc，fabric ①）──────────────┐
   │  分片嵌入表（细粒度锁）+ 版本号 ── 供 Nearline / Online 共享查表           │
   └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么要拆开部署

- **近线（Stage I）与在线（Stage II）负载形态不同**：近线是「写密集、批量尾巴」，在线是「读密集、低时延、高 QPS」。拆开可独立扩容、独立限流、独立灰度。
- **稀疏参数必须独立服务化**：稀疏表可能超过单机内存，且若每个精排实例各自维护一份，会造成内存/存储成倍浪费。独立 PS 使 Nearline/Online 共享同一份稀疏表，按 id 稳定哈希分片，读侧 gather。
- **UserKV 与 owner worker 同节点共存**：用一致性哈希把 user 路由到 owner，使「在线读 KV」命中本地缓存，最大化数据本地性（对应 e2e 决策 D1/D3）。

---

## 3. 计算面线程模型

### 3.1 单机参照实现（Python）

代码位置：`onetrans/serving/dispatcher.py`。要点与 brpc+bthread 一一对应：

| Python 参照 | 生产 brpc+bthread | 说明 |
|---|---|---|
| `Dispatcher.submit` | RPC 入口（bthread handler） | 分配 `req_seq`，按 user 稳定哈希/轮询选 worker |
| `WorkerPool`（N 个独立有界队列） | N 个 bthread worker（M:N） | 每 worker 独立队列，**无全局锁**，QPS 随核扩展 |
| `Future` + `_inflight[req_seq]` | brpc `Controller` 回调 + 序列号 | **req_seq 异步匹配**，支持乱序完成 |
| `OverloadRejected`（队列满即拒） | brpc 背压（限流/拒绝/降级） | 队列有界，满则快速失败，避免雪崩 |

关键设计选择：

1. **消全局锁是吞吐前提**。共享单一 queue + 全局 mutex 会成为高并发瓶颈；改为「每 worker 一条有界队列 + 生产者按稳定哈希派发」，把锁粒度压到单 worker 队列。
2. **req_seq 异步匹配**。请求按单调递增 `req_seq` 编号，worker 完成任意序，`Dispatcher` 经 `Future` 送回对应调用方，天然支持乱序完成、超时取消。
3. **背压是必备手段**。有界队列满即返回 `OverloadRejected`（不阻塞、不丢已入队请求），上层可重试或降级（如 miss 兜底、出候选截断）。

### 3.2 与 LLM 线程模型的差异

本系统第二阶段**非自回归**（M 个候选整批并行打分），无 decode 逐 token 循环，因此不需要 vLLM 的 PD 分离调度（`e2e_design_spec.md` §7）。线程模型重心是「高并发读 KV + 交叉注意力吞吐」，而非「单请求逐 token 时延」。

---

## 4. 独立稀疏参数服务器（PS）

### 4.1 设计要点

- **稀疏表按 id 稳定哈希分片**（N 分片，每分片一把锁）：读写细粒度并发，无全局锁。
- **表版本 `version` 随写递增**：供在线侧做「权重版本化加载 / 失效校验」。
- **未命中回 0 向量，由客户端按 seed 兜底哈希嵌入重建**：对应权重版本化加载的「最差路径」。
- **分片哈希跨语言同构**：Python 侧 `hash64`（`router.py`）与 C++ 侧 Knuth 乘法哈希语义一致，保证同一 feat_id 跨进程稳定落入同一分片。

### 4.2 组件与 wire 契约

| 文件 | 职责 |
|---|---|
| `deploy/ps/embedding_service.proto` | RPC 契约：`Lookup` / `BatchLookup`，字段含 table/ids/dim/weights/version/shard_id |
| `deploy/ps/embedding_server.cc` | brpc + bthread 参考实现：`ShardedEmbeddingTable` + `EmbeddingServiceImpl` |
| `deploy/ps/BUILD` | bazel 构建（依赖 brpc/protobuf/glog/gflags） |
| `onetrans/serving/embedding_ps_client.py` | Python 侧：`ShardedEmbeddingTable` + `LocalEmbeddingPS` + `EmbeddingPSClient` |

### 4.3 客户端数据面

`EmbeddingPSClient`：
- `local` 模式：进程内分片表（数值/并发基准）；
- `remote` 模式：发 brpc RPC 调 `EmbeddingService.Lookup`。

> 当前 `remote` 路径在 Python 侧为**占位**（`NotImplementedError`）：生产环境的 brpc 调用发生在 C++ worker 内，Python 客户端只用于单机基准。详见「实现&现状总结」§5。

---

## 5. KV 零拷贝 UserKV 数据面

### 5.1 拷贝点识别（问题在哪里）

传统路径在 UserKV 读写上存在**可避免的拷贝**：

1. **序列化读侧**：`bytes payload` → `torch.Tensor` 常规做法会逐层 `numpy.frombuffer().copy()` 或 `torch.tensor(list)`，产生一次全量 CPU 拷贝。
2. **进程间传输**：nearline 写、online 读之间若经 socket/UDS 逐字节搬运，再做内存再分配，存在至少两次冗余拷贝。

### 5.2 消除方案

| 环节 | 手段 | 落地点 |
|---|---|---|
| 序列化读侧 | 预分配连续 buffer + `torch.frombuffer` 零拷贝视图 + 逐层 offset 定位 | `serialize.py`（`_view_tensor`、`iterate_tensors`） |
| 序列化写侧 | 预分配 `bytearray` + `ctypes.memmove` 直搬 tensor 底层内存 | `serialize.py`（`serialize`） |
| 进程间传输 | `mmap` 共享内存（`ACCESS_WRITE` 双向零拷贝） | `local_adapter.py`（`mmap_dir` / `_persist`） |

`demo.py` 的 `test_zero_copy` 断言「frombuffer 视图底层缓冲 + mmap 后端读侧零拷贝一致」作为回归。生产侧由 datasystem 的 HBM/DRAM 多级缓存 + 卡间直通（HCCS/RoCE）承接（`detailed_design.md` §3.3）。

---

## 6. 一致性哈希路由 + 元数据/版本失效

### 6.1 一致性哈希路由（数据本地性）

| 文件 | 职责 |
|---|---|
| `serving/router.py` | `JumpConsistentHash`（最小 remap，8→9 桶 remap≈0.116）+ `RingHash`（虚拟节点，动态增删节点稳定）+ `Router` 门面 |
| `serving/sharded.py` | `ShardedKVStore`：按 `user_id` 路由到 shard，`mget` 按 shard 聚合（跨 shard 零复制） |

目标：把 user 请求路由到其 KV **owner**，使读命中本地，避免跨节点搬运。

### 6.2 元数据/版本失效

| 文件 | 职责 |
|---|---|
| `serving/meta_store.py` | `KVPointer`（version/len/checksum/ts/obj_key）+ `LocalMetaStore`（TTL 惰性过期 + LRU 语义） |
| 一致性校验 | Online 读 KV 校验 `header.checksum == pointer.checksum`，不一致触发版本兜底或命中降级（`detailed_design.md` §3.4/§5） |

### 6.3 乐观并发 append

增量 append 携带 `offset` 校验；`offset` 冲突（并发写）即拒绝本次 append，由上层重放/全量 put 兜底。demo 中「正确 offset 接受、冲突拒绝（offset_conflict）」验证该语义。

---

## 7. 动态 batching

| 文件 | 职责 |
|---|---|
| `serving/pipeline.py` | `BatchScheduler`（FIFO 攒批：满批或超时吐批）+ `score_batch`（批量 mget + 打包）、`OnlineWorker` 批处理 |
| `serving/two_stage.py` | `score_ns_batch`（逐层 stack，左 padding 有效掩码，与单请求 `score_ns` 数值等价） |

- **攒批窗口**：`max_wait_seconds` 内未满批也按已攒 ≥1 条返回，保证时延有界。
- **数值等价**：pyramid 每层宽度 `S_l=dims[l]` 对所有 user 恒定，逐层 stack（左 padding）与逐条打分在 B=1 时逐位一致；demo 断言 3 用户攒批与逐条 max|diff| 收敛。

---

## 8. 组件 / 文件清单

```
onetrans/serving/
├── serialize.py            KV 序列化（读侧 frombuffer 零拷贝）
├── local_adapter.py        本地 KV（mmap 共享内存）
├── kv_store.py             KV 统一接口 + LocalKVStore/ShardedKVStore
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
├── embedding_server.cc      PS brpc+bthread 参考实现
└── BUILD                    bazel 构建
```

---

## 9. 边界与未接入项（生产落地缺口）

| 项 | 现状 | 计划 |
|---|---|---|
| PS remote 数据面（Python→brpc） | Python 侧占位 `NotImplementedError` | 由 C++ worker 承载，Python 仅本地基准 |
| Redis 元数据后端 | `kv_store.py` 占位 | 等价 adapter（datasystem 元数据能力可替代） |
| datasystem HBM 直通 | `datasystem_adapter.py` 占位 | 需昇腾 NPU 环境联调 |
| vLLM 自定义 op 移植 | 未开始 | 以 `demo.py` 数值基准为验收门槛 |

> 以上未接入项不是本研究（负载模型研究）的正确性阻断项，属生产落地验收项。详见「实现&现状总结」。