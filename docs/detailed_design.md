# 序列 Transformer 精排系统 详细设计

> 版本：v0.1（详细设计）
> 上游：[端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6）
> 本文件承接概要设计，固化三类细节：**KV 存储接口契约**、**张量契约**、**指标采集点**，并给出逐组件设计。

---

## 0. 文档关系与阅读导航

| 读者 | 关注章节 |
|---|---|
| 存储/平台工程师 | §1 KV 存储接口契约（存储无关 adapter + yuanrong datasystem 映射） |
| 算法/推理工程师 | §2 张量契约、§4 算法→工程映射 |
| 性能/容量工程师 | §6 指标采集点、§7 容量/性能模型 |
| 交付负责人 | §8 实现里程碑 |

约定符号：
- `D=d_model`，`H=num_heads`，`d=head_dim=D/H`，`L=num_blocks`，`Ns=ns_tokens_num`
- `S_len`：某 user 的有效序列 token 数（≤ `max_seq_len`，按需 padding）
- `M`：单请求候选数；`b`：batch 维度

---

## 1. KV 存储接口契约（datasystem adapter，存储无关）

### 1.1 设计原则

1. **存储无关**：上层组件（Nearline/Online Worker）只依赖 §1.3 的**统一逻辑接口**，不感知底层是 yuanrong datasystem、Redis、S3、本地内存还是混合。底层通过 `KVStoreAdapter` 接口适配。
2. **对象（KV）语义优先**：把「一用户、一模型版本」的整套逐层 K/V 作为一个**不可变（append-only）对象**，天然契合 yuanrong datasystem 的 KV 与异构对象两种语义（见 §1.4）。
3. **张量payload、元数据分离**：payload 是连续字节 blob（张量序列化），元数据是结构化小对象；避免把张量塞进制值语义的元数据平面。
4. **版本化 + 增量 append**：写侧只追加 `ΔL` 的 K/V，读侧按版本与校验和做一致性校验（见 §5）。

### 1.2 逻辑对象模型：`UserKVRecord`

```
UserKVRecord
├── header（元数据，结构化）
│   ├── key          : str          # 主键，规范见 §1.6
│   ├── model_version: str          # e.g. "onetrans@2026-08-22"
│   ├── user_id      : str
│   ├── s_len        : int          # 当前有效 S token 数
│   ├── per_layer_len: list[int]    # 每层 S token 数（pyramid 缩后逐层）
│   ├── seq_ts_last  : int          # 最近一次追加的行为时间戳
│   ├── dtype        : str          # "float16" | "bfloat16"
│   ├── layout       : str          # "bshd"（B,S,H,d）
│   ├── checksum     : str          # payload 的 sha256
│   └── created_at   : int
└── payload（连续字节 blob，见 §2.3 序列化）
    └── per-layer K_s^l, V_s^l（l=0..L-1）
```

> 说明：在元数据平面（§3.4）保留的不再是完整 `header` 的副本，而是 `pointer`（version/len/ts/checksum/存储地址），避免双写不一致。`header` 可随 payload 一起存于 datasystem，或只由在线侧按需从 `pointer` + blob 反解析。

### 1.3 统一接口契约（逻辑层，供上层调用）

以下为**存储无关**的接口签名（伪 Python）。上层只 import `kv_store`，不 import 具体后端。

```python
class KVStore(Protocol):
    """User KV 的统一逻辑接口（存储无关）。"""

    # ---- 生命周期 ----
    def connect(self, conf: KVConfig) -> None: ...
    def close(self) -> None: ...

    # ---- 单对象读写（整用户全层）----
    def put(self, rec: UserKVRecord) -> PutResult:
        """整对象写入（全量覆盖某版本）。返回 {accepted, version, checksum}。"""

    def get(self, key: KVKey, *, layers: list[int] | None = None) -> UserKVRecord | None:
        """整对象（或指定层）读取。layers=None 表示读全层。未命中返回 None。"""

    # ---- 增量 append（Nearline 写路径)----
    def append(self, key: KVKey, delta: DeltaKV, expect_from: int) -> AppendResult:
        """在已有对象尾部追加 ΔL 个 token 的 K/V。
        delta 携带 base_version 与 offset；服务端做乐观并发校验（offset==s_len）。
        返回 {accepted, new_s_len, checksum}。"""

    # ---- 批量 ----
    def mget(self, keys: list[KVKey], *, layers: list[int] | None = None) -> list[UserKVRecord | None]: ...

    # ---- 生命周期/失效 ----
    def delete(self, keys: list[KVKey]) -> DeleteResult: ...
    def ttl(self, key: KVKey, ttl_seconds: int) -> None: ...
    def prefetch(self, keys: list[KVKey], *, dest: TransferDest = "hbm") -> list[ObjRef]:
        """（可选）显式预取/迁移到目标介质（HBM/DRAM）。返回零拷贝引用 ObjRef。"""
```

**输入/输出契约要点**：

| 操作 | 输入 | 输出 | 幂等性 | 一致性 |
|---|---|---|---|---|
| `put` | 完整 `UserKVRecord` | `PutResult{accepted, version, checksum}` | 幂等（同 checksum 去重） | Causal |
| `get` | `KVKey(version,user_id)` + 可选层号 | `UserKVRecord｜None` | 只读 | Causal / PRAM |
| `append` | `DeltaKV{key, base_version, offset, tensors}` | `AppendResult{accepted, new_s_len, checksum}` | 非幂等（offset 校验） | Causal |
| `delete` | `list[KVKey]` | 删除计数 | 幂等 | Causal |
| `mget` | `list[KVKey]` | 对齐列表（缺位填 `None`） | 只读 | Causal / PRAM |

`ObjRef` 为零拷贝句柄，语义：调用方在 `ReleaseObjRef` 前持有底层共享内存/显存的合法引用，期间数据不被回收。yuanrong datasystem 用异构对象（`DevPublish/DevSubscribe` + 共享内存）实现。

### 1.4 yuanrong datasystem 语义映射

yuanrong datasystem SDK 提供三类语义：**异构对象（HBM/D2D/零拷贝）**、**KV（共享内存、免拷贝）**、**object（本地对象缓存/Distributed Futures）**。映射规则：

| 数据 | datasystem 语义 | 接口 | 理由 |
|---|---|---|---|
| UserKV payload（DRAM→HBM 迁移、跨卡直通） | **异构对象（hetero）** | `DevPublish / DevSubscribe`，`H2D / D2H` | 推荐 KV 的「读侧大流量 + 跨卡投放」是主瓶颈，需 HBM 直通 |
| UserKV payload（DRAM 驻留、持久化缓存） | **KV** | `kv().set / get([key])` | 免拷贝、可溢出磁盘、可 L2 持久化 |
| 元数据 pointer | **KV** | `kv().set/get` | 小对象高频读写 |
| 冷层持久化（Checkpoint） | **KV + L2 或 object** | `writeMode=WRITE_THROUGH_L2_CACHE`，二级缓存 OBS/SFS | 可靠性与容灾 |

关键映射约束（来自 datasystem 官方限制，写 adapter 时固化）：
- key 仅允许 `[A-Za-z0-9-_!@#%^*()+=:;]`，≤255 字节 → 用 §1.6 的 base64/规范化 key。
- value 大小仅受共享内存限制 → payload 用单 blob 或按层切分，需评估单对象上限（见 §7）。
- 一致性仅 **Causal/PRAM** 两档 → 我们的接口只承诺 Causal（在线读对版本号做校验，见 §5）。
- 可靠性三档 `write_through / write_back / none` → 默认 UserKV 用 `none`（KV 可由 nearline 幂等重放重建），Checkpoint 用 `write_through`。

### 1.5 后端 adapter 实现映射

统一接口 `KVStore` 由 adapter 落地。至少提供三种后端，保证「存储无关」成立：

| 后端 | put/get/mget | append | prefetch | 适用 |
|---|---|---|---|---|
| **yuanrong datasystem** | `kv().set/get`（DRAM）+ hetero（HBM） | `kv().set` + 元数据 offset 校验 | `DevPublish/DevSubscribe` | 生产/集群，主选型 |
| **本地模拟（单卡参照）** | `dict` + `mmap`/tensor 文件 | 内存拼接 | no-op | 单机数值/时延基准（D6 前置） |
| **Redis + S3** | Redis blob（小）/ S3（大） | Lua append | 否 | 冷备/兼容，等价 adapter |

> 单卡参照实现的 adapter 与 yuanrong adapter 共享同一 `KVStore` 接口与同一套张量序列化，保证「先在单卡跑通数值与契约，再切 datasystem 集群」无缝。

### 1.6 key 规范与序列化

- 主键模板：`kv:{model_version}:{user_id}`（user_id 做 url-safe base64）。
- 层粒度（可选，用于 pyramid@read 部分读取或超大对象切分）：`kv:{model_version}:{user_id}:L{l}`；此时 `header` 存整用户，`payload` 按层分片。
- 对象/版本指针 key：`meta:{model_version}:{user_id}` → `{version, s_len, per_layer_len, ts, checksum, obj_key}`。
- payload 序列化：见 §2.3 的 `UserKVRecord` 内存布局，用 `torch.save`（`_use_new_zipfile_serialization=False`）或自定义 header+raw tensor bytes 拼接。**禁止 pickle 任意对象**，仅存 dtype/shape/strides 元信息 + 连续字节。

---

## 2. 张量契约（Tensor contracts）

### 2.1 规格常量（对齐 `onetrans` 默认配置）

| 符号 | 默认值 | 来源 |
|---|---|---|
| `D` | 256 | `d_model` |
| `H` | 4 | `num_heads` |
| `d` | 64 | `head_dim = D/H` |
| `L` | 6 | `num_blocks` |
| `Ns` | 8 | `ns_tokens_num` |
| `max_seq_len` | 100 | `S_len` 上限 |
| `min_seq_len` | 10 | pyramid 末层 S 数 |
| `dims[l]` | 100 → 10（linear/log） | pyramid 缩层序列 |

### 2.2 各边界张量表

默认 `dtype=float16`（训练沿用 fp32，推理量化可 bf16/FP8，见 §4）。布局统一 `bshd = [b, S, H, d]`，进入 `scaled_dot_product_attention` 前转 `bhsd = [b, H, S, d]`。

| # | 边界 | 张量 | 形状 | dtype | 生产者 → 消费者 |
|---|---|---|---|---|---|
| T1 | Nearline 输入 | `S_emb`（融合 token + pos + type emb） | `[1, S_len, D]` | fp16 | S tokenizer → Stage I Block0 |
| T2 | Stage I 逐层缓存 | `K_s^l` | `[1, H, S_len^l, d]` | fp16 | Stage I Block l → KV Store |
| T3 | Stage I 逐层缓存 | `V_s^l` | `[1, H, S_len^l, d]` | fp16 | Stage I Block l → KV Store |
| T4 | Stage I 逐层 S 隐层 | `S_hidden^l`（中间，可丢弃仅缓存 T2/T3） | `[1, S_len^l, D]` | fp16 | Block l → Block l+1 |
| T5 | Online 输入 | `NS_emb`（候选/上下文/用户画像） | `[b, M, Ns, D]` | fp16 | NS tokenizer → Stage II |
| T6 | Stage II 逐层 Q | `Q_ns^l` | `[b, M, Ns, H, d]` | fp16 | NS token-specific 投影 → 交叉注意力 |
| T7 | Stage II 逐层 K/V（NS 自注意） | `K_ns^l / V_ns^l` | `[b, M, Ns, H, d]` | fp16 | NS token-specific 投影 → 交叉注意力 |
| T8 | 交叉注意力 key/value | `K^l = [K_s^l ∥ K_ns^l]` | `[b, M, S_len^l+Ns, H, d]` | fp16 | 读 KV + 在线 NS → attention |
| T9 | Stage II 输出 | `NS_hidden^L`（或 CLS） | `[b, M, Ns, D]` | fp16 | Block L → Head |
| T10 | 最终分数 | `logits` | `[b, M, T]`（T=任务数，单任务 2: pCTR/pCVR） | fp16→fp32 | Head → 排序 |

### 2.3 `UserKVRecord.payload` 内存布局

按 `L` 层顺序紧密拼接，首部放 shape 描述表：

```
payload = <magic:4B> <n_layers:4B> <desc_table: n_layers × (l_i, S_len_i)> <raw_bytes>
raw_bytes = concat(K_s^0, V_s^0, K_s^1, V_s^1, ..., K_s^{L-1}, V_s^{L-1})   # 均按 bshd 连续
每层字节数 = 2 × H × S_len^l × d × sizeof(dtype)
```

- 单用户全层字节量（fp16，S_len=100 恒定，未缩层近似）：
  `L × 2 × H × S_len × d × 2B = 6 × 2 × 4 × 100 × 64 × 2 ≈ 614 KB/用户`（pyramid 实际更小，见 §7）。
- `desc_table` 支持只读指定层（`layers=[...]` 时按偏移抽取，不反序列化整对象）。

### 2.4 增量 append 张量契约（`DeltaKV`）

```
DeltaKV{ key, base_version, offset(=原 s_len), ΔL,
         tensors: {l: (ΔK_s^l [1,H,ΔL,d], ΔV_s^l [1,H,ΔL,d]) for l in 0..L-1} }
```

- 约束：每层 `ΔL` 一致；服务端校验 `offset == header.s_len`，否则返回 `accepted=False`（触发 nearline 全量重建）。
- pyramid@read 说明：近线**缓存全序列**每层 K/V（D2 决策），不做缩层破坏；缩层仅发生在读侧（§4.2）。因此 append 语义始终是「按 offset 追加」，不重算历史层。

---

## 3. 组件详细设计

### 3.1 Nearline Worker（Stage I，prefill 类比）

职责：行为流 `ΔL` → tokenize(S) → 复用旧 K/V 仅算新增位置 → `append` UserKV → 更新元数据。

```
输入：行为事件流（按 user 分区）
步骤：
  1. 读 pointer：get_meta(user) → 若有旧 version 且 offset 匹配，走增量
  2. 增量 encode：S tokenizer 仅对 [s_len, s_len+ΔL) 位置 tokenize
     （causal 下新增 token 的自注意力只依赖历史 token → 只需旧 S_hidden^{L-1} 尾部 + 新增 token）
  3. 逐层算 ΔK_s^l/ΔV_s^l（混合 W_s 共享投影 → 非 softmax 依赖，可并行）
  4. kv_store.append(delta) → accepted?
       - accepted：更新 pointer(version,s_len,ts,checksum)
       - 拒绝（offset 冲突）→ 触发该 user 全量重建（幂等）
输出：UserKV append 成功 + pointer 更新
```

### 3.2 Online Worker（Stage II，decode 类比但并行非自回归）

职责：读 UserKV + 候选特征 → NS 编码 → 逐层交叉注意力 → head 打分。

```
输入：请求(uid) + M 候选
步骤：
  1. kv_store.get(uid, version) 或 prefetch(HBM) 得 ObjRef（零拷贝）
  2. 候选特征服务 + Embedding 查表（fabric ①）→ NS_emb (T5)
  3. Stage II：
       for l in 0..L-1:
         Q_ns^l,K_ns^l,V_ns^l = 逐 token 投影(NS)         # NS token-specific（混合参数化）
         K^l,V^l = [读 K_s^l,V_s^l ∥ K_ns^l,V_ns^l]        # T8
         attention(bhsd) → NS 残差 → mixed FFN(逐 token)
         （可选 pyramid@read：只读 K_s^l 的最新 min(S_len^l, Δ限) 行，节省带宽）
  4. NS_hidden^L → head → logits (T10)
输出：M 个分数
```

### 3.3 KV Adapter / datasystem 客户端封装

- 封装 `DsClient(host, port)` → `kv()/hetero()`，向上暴露 §1.3 的 `KVStore`。
- hetero 路径：`DevPublish`（nearline 把刚算好的 HBM K/V 发布为异构对象）→ `DevSubscribe`（online 申请 HBM 后订阅接收），数据系统用卡间直通（HCCS/RoCE）搬运，避免 CPU 拷贝。
- 备选：DRAM KV（共享内存免拷贝）+ `H2D/D2H` 迁移，用于「近线写 DRAM、在线读时交换到 HBM」的配置。

### 3.4 元数据/版本面（fabric ③）

| 键 | 值 | 写方 | 读方 |
|---|---|---|---|
| `meta:{mv}:{uid}` | `{version, s_len, per_layer_len, seq_ts_last, checksum, obj_key}` | Nearline | Online/路由 |
| TTL/失效 | LRU 或 `ttl()` | 平台 | - |

一致性：Online 读 KV 时校验 `header.checksum == pointer.checksum`；不一致即触发 `get(version)` 兜底或命中降级（陈旧读 + 打点）。

### 3.5 Embedding 数据面（fabric ①）

TorchRec `ShardedEmbedding`（jagged，频率感知分片），被 Nearline（S 特征）与 Online（NS 特征）共享。**只做读侧分片 gather，不建训练完整 PS**（D5）。

### 3.6 编排/路由

- 按 `user_id` 一致性哈希路由到 owner worker（数据本地性：KV 与 owner 共存）。
- Dynamic batching：不同 user 的请求按 padding 打包；M 独立，`b` 为请求数。
- 由 Triton 或 vLLM 原生调度承载（D6），不自研 PD 语义（本系统非自回归，无 decode 逐 token 循环，PD 分离比值不同于 LLM）。

---

## 4. 算法→工程映射

### 4.1 两阶段拆分（对应 LLM prefill/decode）

| 维度 | LLM | 本系统 |
|---|---|---|
| 第一阶段 | prefill（全部 prompt） | Nearline：S 侧编码 + 缓存逐层 K/V |
| 第二阶段 | decode（逐 token 自回归） | Online：NS 编码 + 逐层交叉注意力（**并行，非自回归**） |
| 共享态 | KV cache | User KV（S 侧 K/V） |
| 拆分收益 | 避免重复 prefill | 避免每请求重复编码用户历史 |

- **联系**：两阶段共享同一 backbone 权重与同一 KV 布局；S 侧 K/V 与 NS 无关（严格因果），可安全预计算。
- **差异**：第二阶段无 sequential 依赖，M 候选可整批并行，吞吐瓶颈是「HBM 带宽 + 交叉注意力算力」而非「逐 token 时延」；故无需严格 PD 分池，而是 KV 与 online worker **同节点共存**以最大化数据本地性。

### 4.2 pyramid@read（读侧剪尾）

- Nearline 全量缓存每层 `[H, S_len^l, d]`。
- Online 读侧可只取每层**最新 `min(S_len^l, read_limit)` 行**（tail 保留，最新行为权重大），作为带宽优化消融项（首个实验，见概要 §9）。
- **仓库修正**：`onetrans_block.py` 当前 `s_tokens = z[:, :out_seq_num]` 保留头部（最旧），与论文「保留尾部」相反；需改为 `s_tokens = z[:, s_len - out_seq_num : s_len]` 并配左 padding。

### 4.3 混合参数化落地

- S token 全共享一套投影（`W_s`、`network_s`）；NS 逐 token 独立（`W_ns_list[i]`、`networks_ns_list[i]`）。
- vLLM 无法原生表达「逐 token 权重」，策略：先单卡参照实现固化数值基准，再做 vLLM 自定义 op 移植（D6）。

---

## 5. KV 生命周期、版本化、失效

- **版本**：`model_version` 变化（重训发布）→ 新建 key 前缀，旧版本延迟退役（TTL）。
- **append vs put**：增量走 `append`（offset 校验）；`model_version` 或 `s_len` 不一致时走 `put` 全量重建。
- **失效**：TTL=活跃窗口（如 7d）；LRU 淘汰（热 KV 多副本跨节点，LRU 本地副本）。
- **重建可靠**：UserKV 可用「原始行为日志 + Stage I」幂等重放，故 K/V 层默认 `writeMode=none`，Checkpoint 用 `write_through`。

---

## 6. 指标采集点（Metric collection）

埋点统一经 `onetrans/utils/metrics.py` 的 `logq` 出口，走 OpenTelemetry + Prometheus。采集点以 `组件.阶段.子操作` 命名。

### 6.1 时延（Histogram，单位 ms，标签 `component/stage/pct`）

| 指标 | 采集点 |
|---|---|
| `nearline.encode_stage1` | Nearline：S tokenize + L 层编码 |
| `nearline.append_kv` | datasystem `append` 往返 |
| `online.kv_get` | datasystem `get`/`prefetch` |
| `online.embed_gather` | Embedding fabric 查表 |
| `online.encode_stage2` | NS 编码 + 逐层交叉注意力 |
| `online.e2e_p50/p99/p999` | 请求入口→返回 |
| `kv.h2d / kv.d2h / kv.d2d` | 迁移原语 |

### 6.2 吞吐（Gauge/Counter）

| 指标 | 采集点 |
|---|---|
| `online.qps` | 排序入口 |
| `online.candidate_throughput`（cand/s、token/s） | Stage II 头 |
| `nearline.events_ingested`（event/s） | 行为流入口 |
| `kv.tput_read/write`（GB/s） | adapter |

### 6.3 资源（Gauge，标签 `host/gpu`）

| 指标 | 采集点 |
|---|---|
| `gpu.sm_util`（%）、`gpu.hbm_bw`（GB/s）、`gpu.hbm_cap`（%） | DCGM |
| `dram`/`ssd` 用量、`net.rdma_bw` | 节点 exporter |
| `kv.cap_used`（GB）、`kv.obj_cnt` | datasystem worker 侧 |

### 6.4 KV 存储（datasystem，Gauge/Counter）

| 指标 | 采集点 |
|---|---|
| `kv.hit_rate`、`kv.miss` | adapter.get |
| `kv.spill_bytes_in/out` | 溢出路径 |
| `kv.replica_hit`（跨节点本地副本命中） | adapter 统计 |
| `kv.version_conflict`（append 拒绝计数） | append 返回 |
| `kv.checksum_mismatch`（读侧校验失败，触发兜底） | online 读 |
| `kv.replace_full`（全量重建计数） | nearline |

### 6.5 质量/正确性（离线，非线上直采）

| 指标 | 采集点 |
|---|---|
| `offline.auc / gauc / ndcg` | 评测集 |
| `offline.online_consistency`（两阶段 vs 单前向数值 max-diff） | 反序列化/切分正确性回归 |
| `offline.kv_on_hit_score_delta`（陈旧 KV 对分影响） | 降级实验 |

### 6.6 埋点位置总表（与代码对齐）

| 组件 | 文件/位置 | 埋点 |
|---|---|---|
| Nearline | `run/main.py` 推理分支、`builder.py` | encode_stage1、events_ingested |
| Online | Stage II forward、head | encode_stage2、logits 输出 |
| Adapter | `ext/*/datasystem_client.py`（新增） | kv_get/kv_hit/version_conflict/checksum |
| Embedding | TorchRec gather | embed_gather |
| 资源 | 不侵入代码，DCGM/exporter | §6.3 |

---

## 7. 容量/性能估算模型

**参数化模型**（先于任何集群部署，作初步量级）：

- 单用户 KV 容量（fp16，pyramid 近似）：
  `C_kv = Σ_{l=0}^{L-1} 2 · H · S_len^l · d · 2B`，取 `S_len^l = dims[l]` 线性 100→10，L=6：
  `C_kv ≈ 2·4·64·2 · Σ dims[l] = 1024 · (100+82+64+46+28+10) ≈ 1024·330 ≈ 338 KB/用户`。
- 冷启动/热读带宽：`R × C_kv`（R=读 QPS 用户数）× 每层 `read_limit/S_len` 剪尾比。
- 交叉注意力算力（Stage II）：`O(M · Ns · Σ S_len^l · D)`，与 KV 读取带宽共同决定 online 每卡 QPS 上限。
- 三个可调杠杆：`pyramid@read` 剪尾比、dtype 量化（fp16→bf16/FP8）、KV 与 worker 同节点数据本地化。

> 数值为量级占位，最终以 §6 埋点实测校准。详细建模与基准方法见概要 §5/§8 与后续 Track B 选型文档。

---

## 8. 实现里程碑

| 阶段 | 内容 | 出口 | 依赖 |
|---|---|---|---|
| M0 | 修正 pyramid 方向 + 固化两阶段数值基准 | 单前向 vs 两阶段 max-diff < ε | `onetrans_block.py` 修正 |
| M1 | `KVStore` 接口 + 本地模拟 adapter + 序列化 | 契约单测、roundtrip 一致性 | §1/§2 |
| M2 | datasystem adapter（KV + hetero） | 集群读写基准 | yuanrong SDK 部署 |
| M3 | 指标埋点 + 端到端负载实验 | §6 全指标可采 | M1/M2 |
| M4 | DeepFM/DCNv2 等约束公平对比 | Pareto 前沿 | Embedding 同 seam |

---

## 9. 与概要设计的对应关系

| 概要章节 | 详细章节 |
|---|---|
| §4 三 fabric | §3.4 / §3.5 |
| §5 ADR D3/D6 | §1.4 dataSystem 映射 / §4.3 |
| §6 组件清单 | §3 组件详细设计 |
| §7 数据流 | §3.1 / §3.2 |
| §8 技术选型 | §1.4 / §3.3 |
| §9 风险（datasystem 适配 / pyramid@read） | §1.5 / §4.2 / §7 |