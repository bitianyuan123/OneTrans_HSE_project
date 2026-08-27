# 第二阶段详细设计：工程可用（Engineering-Ready）

> 版本：v1.0
> 文档类别：**③ 现状 & 差距 · 阶段执行设计**
> 上游：[detailed_design.md](./detailed_design.md)（② 目标设计：§7.1.2 组件视图、§7.4 并发模型、§7.8 可靠性/可观测性、§4.5 PS 契约）；[gap_analysis.md](./gap_analysis.md)（G1~G14 差距清单、M6/M7 设计基型）
> 阶段定位：Phase 1（单机参照实现 + M5 正确性收口）**已完成**；**本文是 Phase 2「工程可用」的执行设计**；Phase 3（C++ 生产化 M8a~c、G12/G14）不在本文范围。
> 代码基线：`feat/onetrans-e2e-serving` @ 90063ae。本文所有「既有落点」均为真实存在的文件/签名。

---

## 0. 「工程可用」的验收定义（DoR）

第二阶段完成，当且仅当满足五个支柱（E1~E5）与端到端验收场景（§8）：

| # | 支柱 | 含义（可证伪） | 对应差距 |
|---|---|---|---|
| E1 | 真实输入闭环 | nearline/online 消费**原始特征**（行为事件 / 候选特征 JSON），embedding 查表经 PS（fabric ①），而非预 tokenize 张量 | G7 |
| E2 | 故障韧性 | 超时 / 重试 / 熔断 / 限流 / 健康检查 / 优雅停机（drain）全部具备且可注入故障验证 | G4 |
| E3 | 可观测 | 指标分桶直方图 + Prometheus 文本导出；req_id/trace_id 结构化日志贯穿 ingress→worker→后端 | G5 |
| E4 | 可运维 | 配置文件驱动启动，端点/版本零硬编码；模型版本灰度路由（双版本并存打分） | G9 |
| E5 | 可回归 | pytest 测试矩阵 + 金样本 + CI；`demo.py` 全部既有断言固化进测试 | G13（升级 P1） |

**明确不在本阶段**：C++ 热路径（G6/M8）、redis/HBM 直通（G12）、局部性能（G14）。Phase 2 的产出形态仍是 **Python 单机工程形态**——它是 [detailed_design.md §7.1.2](./detailed_design.md)「验证基准组件视图」的完整落地；性能结论一律以 Phase 3 的 C++ 生产实现为准。

---

## 1. 差距收敛：本阶段做什么、不做什么

| 差距 | 级别 | 处置 | 理由 |
|---|---|---|---|
| G4 可靠性四件套 | P1-High | **本阶段（T2）** | 无它则任一后端抖动即雪崩，非「可用」 |
| G5 可观测性 | P1-Mid | **本阶段（T3）** | 负载实验与线上排障的前提 |
| G7 热路径接线 | P1-High | **本阶段（T1）** | 不接线则系统只能吃人造张量，一切实验输入不真实 |
| G9 服务发现/灰度 | P1-Mid | **本阶段（T4）** | 端点硬编码 + 无灰度 = 不可运维 |
| G13 测试/CI | 原 P2 | **升级 P1，本阶段最先做（T0）** | G7 要改入口签名、T2 要改 dispatcher——没有回归基座，每一步都在裸奔 |
| G6 C++ 移植 | P1-Mid | 延后 Phase 3 | 性能腿而非可用腿；依赖 vLLM 自定义 op/算子环境，工程量大 |
| G12/G14 | P2 | 延后 Phase 3 | 非阻断 |

**G13 升级的依据**：第二阶段的每一轨都会改动既有热路径（T1 改 worker 入口、T2 改 Dispatcher/adapter、T3 改 metrics 内部）。先固化「当前正确行为」为测试，是后续所有改动的安全网——这与 detailed_design §6.1「先数值、后工程」的策略一脉相承。

---

## 2. 阶段架构总览

第二阶段完成后的组件视图（Python 工程形态；对应 ② 类文档 §7.1.2 的完整落地）：

```
      行为事件(JSON/回放源)                      打分请求(JSON, 含候选特征)
            │                                            │
┌───────────▼──────────────────────────────────────────── ▼──────────────────┐
│ HTTP Ingress（stdlib ThreadingHTTPServer —— server.py, 新）                    │
│  POST /ingest   POST /score   GET /healthz   GET /metrics                      │
│  职责：req_id/trace_id 注入 · 令牌桶限流 · draining 时拒绝 · JSON↔特征 装配      │
└───────────┬────────────────────────────────────────────┬─────────────────────┘
            │ Dispatcher.submit（deadline_s）              │
┌───────────▼──────────────────────────────────────────── ▼────────────────────┐
│ Dispatcher + WorkerPool（既有：req_seq 异步匹配/背压/hash 路由）  ← dispatcher.py │
│  新增：deadline reaper（过期 Future 以 TimeoutError 结算）· drain 优雅停机        │
└───────────┬───────────────────────────┬──────────────────────────────────────┘
        Nearline 路径                Online 路径（VersionRouter 先选版本）
            ▼                                   ▼
┌─────────────────────────┐      ┌──────────────────────────────────────────┐
│ ServingFrontend（新）     │      │ ServingFrontend（新）   ← frontend.py      │
│ UserEvent                │      │ ScoreFeatures                             │
│  → PS 查表(item)          │      │  → PS 查表(user/item/artist/album)          │
│  → tokenize → (s_emb)    │      │  → dense 编码 → tokenize → (ns_emb)         │
└───────────┬─────────────┘      └──────────────────┬─────────────────────────┘
            │ NearlineWorker.ingest（既有）          │ OnlineWorker.score（既有）
            │ encode_s → serialize → put            │ kv.get → score_ns
            ▼                                       ▼
┌──────────────────────┐   熔断/重试/超时(kv)   ┌──────────────────────┐
│ KVStore（既有协议）    │◀───────────────────▶ │ EmbeddingPSClient     │
│ local / datasystem    │                      │ local / remote(brpc)  │
└──────────────────────┘                      └──────────────────────┘
横切：ServingMetrics（分桶直方图+Prometheus，改）· logctx（req_id/trace_id，新）
     · HealthChecker（新）· Config/Registry/VersionRouter（新）
```

**新增/改动模块清单**：

| 模块 | 轨 | 动作 | 职责 |
|---|---|---|---|
| `onetrans/serving/frontend.py` | T1 | 新增 | `ServingFrontend` 协议 + `PSFrontend` 实现 + 原始输入数据类 |
| `onetrans/serving/resilience.py` | T2 | 新增 | `CircuitBreaker` / `TokenBucketLimiter` / `with_retry` / `HealthChecker` |
| `onetrans/serving/server.py` | T2/T3 | 新增 | HTTP ingress（/ingest /score /healthz /metrics，仅标准库） |
| `onetrans/serving/logctx.py` | T3 | 新增 | `ReqContext`（contextvar）+ 结构化 JSON 日志 |
| `onetrans/serving/registry.py` | T4 | 新增 | `ServiceRegistry` 协议 + `StaticRegistry` + `VersionRouter` |
| `onetrans/serving/config.py` | T4 | 新增 | `ServingConfig` + `load_config`（JSON 配置装配） |
| `onetrans/serving/metrics.py` | T3 | 改 | `_Histogram` 分桶化 + `to_prometheus` 文本导出 |
| `onetrans/serving/dispatcher.py` | T2 | 改 | deadline reaper / drain / `OverloadRejected.reason` |
| `onetrans/serving/pipeline.py` | T1 | 改 | `ingest_features` / `score_features`（frontend 注入，旧签名不动） |
| `tools/publish_weights.py` | T1 | 新增 | checkpoint → PS 四表发布 + dense/tokenizer 装载文件导出 |
| `tests/` + CI | T0 | 新增 | 测试矩阵 + 金样本 + GitHub Actions |

**与生产形态的映射**：HTTP ingress 是 stdlib 参照物（验证/冒烟/负载入口），生产对应 brpc Server（bthread-per-RPC）；`CircuitBreaker`/`TokenBucketLimiter` 对应 brpc 生态的熔断限流组件；`VersionRouter` 对应注册中心的灰度路由。语义先在 Python 固化并由测试锁定，Phase 3 映射到 C++。

---

## 3. T0：测试与回归基座（G13）

### 3.1 目标

把 `demo.py` 的脚本断言升级为**可重复、可进 CI、分文件可定位**的测试矩阵；为 T1~T4 的每一步改动提供回归安全网。

### 3.2 测试结构

```
tests/
├── conftest.py                # 固定 torch seed；demo 同构小配置 fixture
│                              #   （D=128, H=4, L=4, max_seq=50, Ns=8, fp16）
└── serving/
    ├── test_equivalence.py    # 两阶段 vs 单前向（valid_len 5/23/37/50 × M 1/5）→ V1
    ├── test_serialize.py      # roundtrip / 零拷贝 / s_len·per_layer_len 固化 → V2（G1 回归）
    ├── test_kv_store.py       # put 幂等 / append offset·CAS·layer 冲突 → V3（G2 回归）
    ├── test_router.py         # worker_for==Router.route / remap 受控 → V4（G3 回归）
    ├── test_pipeline.py       # miss 全零 / miss·hit 混批 / 攒批等价 → V5（G8 回归）
    ├── test_dispatcher.py     # 并发完成 / req_seq 匹配 / 背压拒绝
    ├── test_embedding_ps.py   # Knuth 分片稳定 / 多表 / seed 兜底 / 版本递增
    ├── test_frontend.py       # T1：前端接线数值等价（金样本）
    ├── test_resilience.py     # T2：熔断状态机 / 限流 / 超时 / drain
    ├── test_observability.py # T3：分桶统计 / Prometheus 文本 / logctx 贯穿
    └── test_registry.py       # T4：静态注册 / VersionRouter 权重分桶稳定性
```

- **验证矩阵映射**：`test_*` 文件名后标注的 V1~V5 对应 [model_design.md §6](./model_design.md) 的验证矩阵编号，验收口径一致。
- **fixture**：`conftest.py` 提供 `backbone` / `runner` / `store(local)` / `tokenizer` / `frontend` 等 fixture，全部固定 seed；测试之间无共享可变状态。

### 3.3 金样本（golden fixtures）

- `tests/serving/gen_golden.py`：以固定 seed 生成并落盘——输入（`UserEvent`/`ScoreFeatures` 的 JSON）、逐层 K/V 的逐张量 sha256、最终 logits；
- `test_frontend.py` 断言「前端产出 == 金样本」（max|diff| < 1e-4，容差分级见 model_design §6.2）；
- **重生成必须显式**：模型结构/ tokenizer 配置变化时需人工执行 `gen_golden.py` 并 review diff，防止「测试跟着错误实现漂移」。

### 3.4 CI

`.github/workflows/ci.yml`：torch CPU 版 + `pytest tests/ -q`；push / PR 触发。无 GPU 依赖（`demo.py` 同样纯 CPU 可跑，见 detailed_design §5.3）。

### 3.5 验收

- `pytest tests/` 全绿；`demo.py` 保留为冒烟脚本（行为不变）。
- 故意注入一个回归（如临时改掉 append 的 CAS 校验）→ CI 能红。

---

## 4. T1：真实输入接线（G7）

### 4.1 目标

把「行为流→查表→编码→写 KV」「候选特征→查表→打分」两段入口接成完整链路，同时**保持 worker 与后端解耦（接口注入，不硬编码 remote）**——与 gap_analysis §3.7 的设计基型一致，本文细化到数据类与文件级。

### 4.2 原始输入数据模型（`frontend.py`）

对齐训练侧 `YambdaEmbedder.forward` 的输入语义（S 侧仅 item 序列；NS 侧 = dense + uid + item + artist/album 五组）：

```python
@dataclass
class UserEvent:                  # nearline 输入（行为流一条，时间升序）
    user_id: str
    item_ids: list[int]           # 历史行为 item 序列
    timestamps: list[int]

@dataclass
class CandidateFeatures:          # 单个候选
    item_id: int
    artist_ids: list[int]         # multivalent（mean-bag 语义）
    album_ids: list[int]
    dense: list[float] | None     # item/ui 级稠密（与用户级拼接，顺序=DENSE_COLUMNS）

@dataclass
class ScoreFeatures:              # online 输入（一次打分请求，M 个候选）
    user_id: str
    uid_sparse: int               # 用户稀疏 id（NS 侧 uid 查表）
    dense: list[float]            # 用户/上下文级稠密特征
    candidates: list[CandidateFeatures]
```

### 4.3 前端协议与实现

```python
class ServingFrontend(Protocol):
    def encode_s(self, event: UserEvent) -> tuple[Tensor, Tensor]:
        """原始事件 → PS 查表 → tokenize → (s_emb [1,S0,D], s_mask [1,S0])"""

    def encode_ns(self, feats: ScoreFeatures) -> tuple[Tensor, Tensor]:
        """候选特征 → PS 查表 → tokenize → (ns_emb [M,Ns,D], ns_mask [M,Ns])"""
```

`PSFrontend`（`frontend.py`，本阶段唯一实现）的组合：

| 组件 | 来源 | 职责 |
|---|---|---|
| `ps: dict[str, EmbeddingPSClient]` | 注入（item/user/artist/album 四个 client） | 稀疏查表（local 或 remote 由 client 决定，worker 不感知） |
| `dense_encoder: PiecewiseLinearEncoder` | checkpoint 装载 | dense 分箱编码（进程内，无稀疏查表） |
| `tokenizer: OneTransTokenizer` | checkpoint 装载 | `encode_s`/`encode_ns`（mlps/type/pos/RMSNorm 全部训练态权重） |
| `max_seq_len: int` | 配置 | 截断与 padding |

**`encode_s` 流程**（语义逐条对齐 serving 约定）：

1. `item_ids` 截尾保留**最新** `max_seq_len` 条（`timestamps` 同步），**左 padding** 对齐到 `S0`（有效 token 在尾部——两阶段引擎与掩码的全部既有约定，见 detailed_design §2.3）；
2. PS 查表 `item` 表 → `[1, S0, d]`；
3. `tokenizer.encode_s([seq_emb], [s_mask], [padded_ts])` → `(s_emb, s_mask)`（含 mlp/type emb/pos emb/RMSNorm）。

**`encode_ns` 流程**：

1. `uid_sparse` → user 表 → `[M, d]`（同一 user 广播 M 行）；
2. 候选 `item_id` → item 表 `[M, d]`；`artist_ids/album_ids` → 对应表 + **mean-bag 池化**（对齐训练侧 `embed_multivalent` 的 `embedding_bag(mode="mean")` 语义）；
3. `dense`：用户级 + 候选级按 `run/config.DENSE_COLUMNS` 固定顺序拼接 → `dense_encoder` → `[M, dense_dim]`；
4. `tokenizer.encode_ns([dense, uid, item, artist, album])` → `(ns_emb, ns_mask)`（组顺序与训练侧 `YambdaEmbedder.ns_group_dims` 一致，不得调换）。

### 4.4 worker 扩展（旧签名不动）

```python
class NearlineWorker:
    def __init__(self, runner, store, metrics=None, dtype="float16",
                 frontend: ServingFrontend | None = None) -> None: ...

    def ingest_features(self, event: UserEvent, model_version: str) -> PutResult:
        s_emb, s_mask = self.frontend.encode_s(event)          # 计入 metrics: nearline.frontend
        return self.ingest(s_emb, s_mask, event.user_id, model_version,
                            seq_ts_last=event.timestamps[-1] if event.timestamps else 0)

class OnlineWorker:
    def __init__(self, runner, store, metrics=None,
                 frontend: ServingFrontend | None = None) -> None: ...

    def score_features(self, feats: ScoreFeatures, model_version: str) -> Tensor:
        ns_emb, _ = self.frontend.encode_ns(feats)            # 计入 metrics: online.frontend
        return self.score(feats.user_id, model_version, ns_emb)
```

- `frontend=None` 时保持现状（消费预 tokenize 张量）——`demo.py` 与既有测试零改动；
- `model_version` 显式传参（T4 灰度时由 `VersionRouter.pick` 决定，见 §7.4）。

### 4.5 权重发布工具（`tools/publish_weights.py`）

打通「训练 checkpoint → serving 消费面」：

1. 读训练 checkpoint（`run/train.py:load_training_checkpoint` 的产物）；
2. 拆 embedder 四张稀疏表 → `ps.set_many("emb:{model_version}:{item|user|artist|album}", ids, weights)`（表名规范 = `emb:{mv}:{role}`，与 PS 多表注册中心 G11 对齐）；
3. dense/tokenizer/backbone 装载文件路径写入发布清单（`release.json`：各表名、checksum、checkpoint 路径）；
4. **发布校验**：随机抽样 N 个 id 回读 PS，与 checkpoint 权重逐位比对；输出摘要。

### 4.6 验收

- **数值等价（核心）**：同 checkpoint、同权重发布到 PS 后，「`UserEvent/ScoreFeatures` → `PSFrontend` → `ingest/score`」与「训练侧 `YambdaEmbedder` + `OneTransTokenizer` → `ingest/score`」产出的 K/V 与 logits **逐位一致**（金样本断言，< 1e-4）；
- 前端在 `EmbeddingPSClient.local()` 与 remote（占位换真实）下行为一致，无 host 硬编码（remote 端点来自 T4 配置）；
- miss 语义：PS 未发布的 id 走 seed 兜底（确定性，同 id 同向量）。

---

## 5. T2：可靠性四件套（G4）

### 5.1 超时（三层，全部落地才验收）

| 层 | 机制 | 落点 |
|---|---|---|
| 调用方 | `fut.result(timeout=deadline)`（既有能力，ingress 统一使用） | `server.py` |
| 编排层 | `Dispatcher.submit(..., deadline_s=...)`：`_inflight` 记录 `(Future, deadline_monotonic)`；**reaper 守护线程**每 100ms 扫描，过期 → `fut.set_exception(TimeoutError)` + pop（`_complete` 已有 `fut.done()` 判定，晚到结果自然丢弃） | `dispatcher.py` |
| 数据面 | `YuanrongKVStore`/`EmbeddingPSClient(remote)` 调用携带 `timeout_ms` 透传 SDK/brpc Controller | 两个 adapter |

### 5.2 重试与幂等（`resilience.py`）

```python
def with_retry(fn, *, attempts=2, backoff_s=0.05,
               retry_on=(TimeoutError, TransientError)):
    """仅用于幂等读：kv.get/mget、ps.lookup。写路径不走此函数。"""
```

写路径幂等性沿用既有契约，不新增机制：`put` 按 checksum 幂等覆盖；`append` 非幂等靠 offset+CAS 拒绝重放（G2 已固化），冲突由上层全量 `put` 重建。

### 5.3 熔断器（`resilience.py`）

```python
class CircuitBreaker:
    """按后端实例（kv / ps 各一个）的错误率熔断。"""
    def __init__(self, name: str, *, window_s=10.0, min_samples=20,
                 error_rate=0.5, cooldown_s=5.0): ...
    def allow(self) -> bool
    def record(self, ok: bool)
```

状态机：`CLOSED --错误率超阈值--> OPEN --cooldown 到期--> HALF_OPEN --探测成功--> CLOSED / --失败--> OPEN`。滑动窗口 = `deque[(ts, ok)]` + 锁。OPEN 期间对应该后端的调用直接失败（快速），打点 `circuit.open`。熔断器实例挂在 adapter 调用点与 `OnlineWorker`/`NearlineWorker` 的后端访问之间（组合，不改 KVStore 协议）。

### 5.4 限流（`resilience.py`）

```python
class TokenBucketLimiter:
    def __init__(self, rate_qps: float, burst: int): ...
    def try_acquire(self, n: int = 1) -> bool
```

挂在 ingress 入口（`POST /score`/`/ingest`）：超限 → HTTP 429 + `ingress.rate_limited` 打点（与编排层 `OverloadRejected` 双层背压互补：接入层限 QPS，编排层限队列水位）。

### 5.5 健康检查（`resilience.py` + `server.py`）

```python
@dataclass
class Check: name: str; ok: bool; detail: str

class HealthChecker:
    def __init__(self, probes: list[tuple[str, Callable[[], Check], bool]])  # (名, 探针, critical)
    def check(self) -> tuple[bool, dict]      # 任一 critical 失败 → unhealthy
```

| 探针 | 判定 | critical |
|---|---|---|
| `kv` | `store.get` 哨兵 key 可达（miss 也算 OK——探测「可达」非「命中」） | 是 |
| `ps` | `ps.version()` 可达 | 是 |
| `queue` | `pool.pending() / capacity` 低于阈值（默认 0.9） | 否 |
| `circuit` | 无熔断器处于 OPEN | 否 |

`GET /healthz` 返回 JSON `{status, checks}`，unhealthy 时 HTTP 503。

### 5.6 优雅停机（drain & wait）

```python
class WorkerPool:
    def stop(self, *, drain_timeout_s: float = 30.0) -> bool:
        # 1) 置 _draining=True：此后 submit 在入队前检查 → OverloadRejected(reason="draining")
        # 2) 向每个队列尾部投 _STOP 哨兵（哨兵排在既有请求之后 → 天然 drain 语义）
        # 3) join(timeout)；超时仍有存活 worker → 返回 False（上层记日志）
```

`OverloadRejected` 扩展 `reason: str = "queue_full"`（新增 `"draining"`，向后兼容）。`server.py` 的 `stop()` 顺序：**停止 accept → dispatcher drain → store.close()**。

### 5.7 HTTP ingress（`server.py`，仅标准库）

| 路由 | 方法 | 语义 |
|---|---|---|
| `/score` | POST | `ScoreFeatures` JSON →（限流→`Dispatcher.submit`→`score_features`）→ `{"logits": [[...]]}` |
| `/ingest` | POST | `UserEvent` JSON → `ingest_features` → `{"accepted": bool}` |
| `/healthz` | GET | §5.5 |
| `/metrics` | GET | §6.2 Prometheus 文本 |

线程模型：`ThreadingHTTPServer` 每请求一线程（与生产 bthread-per-RPC 语义同构，GIL 限制下仅作验证/冒烟/负载入口，非生产入口）。入站 `X-Req-Id`/`X-Trace-Id` header 继承（缺省生成）。

### 5.8 验收

- 注入慢 PS（`LocalEmbeddingPS` 增加测试专用 `inject_delay(ms)` 钩子，仅 tests 可见）：请求在 deadline 内返回 `TimeoutError`，不无限等待；
- 错误率达阈值 → 熔断 OPEN（`/healthz` 的 circuit 检查可见）→ 恢复 → HALF_OPEN → CLOSED 自愈；
- 超限 QPS 压 ingress → 429 + 打点；
- 停机：drain 后 `inflight()==0`、队列空、已在队请求全部完成（结果不丢）。

---

## 6. T3：可观测性（G5）

### 6.1 分桶直方图（改 `metrics.py`）

- `_Histogram` 内部改为**固定对数桶** `+Inf` 兜底：`_BUCKETS_MS = (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)`，存 `counts[]` + `_sum` + `_count`（**内存有界**，替换现全样本 list——长跑不再膨胀）；
- `snapshot()` 对外字段**不变**（`{name}_p50/_p99/_count`，percentile 从桶内插值），`demo.py`/既有测试零改动；
- `ServingMetrics` 三类原语（`timing/count/gauge`）与 `MetricsSink` 协议不变。

### 6.2 Prometheus 文本导出（`metrics.py` 新增 `to_prometheus`）

手写 exposition 格式，零新依赖：timing → `onetrans_{name}_bucket{le="..."} / _sum / _count`；counter → `onetrans_{name}_total`；gauge 原样。挂 `GET /metrics`。命名沿用 `组件.阶段.子操作` 约定（`.` → `_`）。

### 6.3 结构化日志与 trace 贯穿（`logctx.py`）

```python
@dataclass(frozen=True)
class ReqContext:
    req_id: str; trace_id: str
    user_id: str = ""; model_version: str = ""

@contextmanager
def bind(**fields) -> Iterator[None]: ...      # contextvar 传播（worker 线程内自动携带）
def current() -> ReqContext | None: ...
def log_event(level: str, event: str, **fields) -> None:
    # 单行 JSON（ts + ctx + event + fields）→ stderr；无第三方依赖
```

- **注入点**：ingress 生成/继承 req_id/trace_id → `Dispatcher.submit` 存入 `Request` → worker 线程 `bind()` 后执行 handler → `log_event("kv.hit"/"kv.miss"/"circuit.open"/"overload"/"drain"...)`；
- **trace（最小可用）**：span 即结构化日志事件（`event="span", op="kv_get", dur_ms=...`），同一 `trace_id` 可串联 Nearline→Online→PS；OTel/SpanSink 导出器后置 Phase 3（协议口已留在 `MetricsSink` 同层）。

### 6.4 验收

- `/metrics` 可被抓取且桶计数与注入的请求量一致；
- 任一请求可从日志中按 `req_id` 关联出完整链路（ingress → worker → kv/ps）；
- `snapshot()` 语义兼容：`demo.py` 输出不变（p50/p99 字段名不变）。

---

## 7. T4：配置化、服务发现与灰度（G9）

### 7.1 配置（`config.py`）

`ServingConfig`（dataclass）+ `load_config(path)`（JSON）。覆盖：模型版本清单（含 checkpoint/表名/灰度权重）、kv 后端与参数、ps 端点与模式、compute（workers/队列/mode）、batch、resilience（timeout/retry/circuit/rate_limit）、server（port）。**启动即 `load_config`，代码内不再出现任何默认 host/port**（`EmbeddingPSClient`/`YuanrongKVStore` 的默认端点参数仅保留为构造缺省，装配层一律显式传入）。

### 7.2 服务发现（`registry.py`）

```python
@dataclass
class Endpoint: host: str; port: int

@dataclass
class ModelRelease:
    model_version: str
    checkpoint_dir: str
    ps_tables: dict[str, str]        # role → 表名（item/user/artist/album）
    target_weight: float = 1.0        # 灰度权重（同层归一化）

class ServiceRegistry(Protocol):
    def resolve(self, name: str) -> list[Endpoint]: ...
    def get_model(self, model_version: str) -> ModelRelease: ...
    def list_models(self) -> list[ModelRelease]: ...

class StaticRegistry:                 # 默认实现：从 ServingConfig 装配，零外部依赖
```

- `StaticRegistry` 满足单机/测试；etcd/consul 实现同协议后置替换（`watch()` 变更订阅为可选扩展）；
- `EmbeddingPSClient`/`YuanrongKVStore` 构造时从 `resolve()` 取端点，替换 `127.0.0.1` 硬编码。

### 7.3 灰度路由（`registry.py`）

```python
class VersionRouter:
    def __init__(self, releases: list[ModelRelease]): ...   # target_weight 归一化为累积区间
    def pick(self, user_id: str) -> str: ...
```

- `pick`：`hash64(user_id) / 2^64 ∈ [0,1)` 落入累积权重区间 → **同一 user 稳定路由同一版本**（灰度期间请求黏着，不抖动）；调整权重只影响边界 user；
- 复用 `router.py` 的 `hash64`（跨进程稳定，与 KV 分片同源）。

### 7.4 多版本并存装配

- **Online**：`OnlineWorker` 持 `runners: dict[mv, TwoStageRunner]` 与 `frontends: dict[mv, PSFrontend]`（每版本一份权重）；`score_features` 先 `pick(user_id)` 再选 runner/frontend；**KVKey 天然携带 model_version（既有设计）→ 双版本 KV 共存、互不污染**；
- **Nearline**：按「当前主版本 + 灰度版本」双写（或仅主版本，由配置开关 `nearline.mirror_versions` 控制）；
- 内存约束：Phase 2 限 **≤2 个并存版本**（backbone+tokenizer 权重 ×N）；更多版本属 Phase 3（C++ 权重共享）。

### 7.5 验收

- 零外部依赖时（纯配置文件）`pytest`/`demo.py` 全量通过；
- 双版本（w=0.5/0.5）下：同一 user 稳定命中同一版本；两版本 KV 记录按 `KVKey.model_version` 隔离可并存读；
- 改配置文件中的端点/权重，重启后生效，代码零改动。

---

## 8. T5：阶段端到端验收（DoR 出口）

一条剧本跑通 E1~E5（固化为 `tests/serving/test_e2e_phase2.py`，标记 `e2e`）：

| 步 | 动作 | 断言 |
|---|---|---|
| 1 | `load_config` 启动：local KV + local PS + N worker + ingress | 进程拉起，无硬编码端点 |
| 2 | `publish_weights`：checkpoint → PS 四表 | 抽样回读逐位一致 |
| 3 | `POST /ingest`（原始行为事件） | KV 写入；`per_layer_len` 与事件长度一致 |
| 4 | `POST /score`（原始候选特征） | logits 与金样本一致（< 1e-4） |
| 5 | 注入 PS 延迟 500ms | deadline 内 TimeoutError；错误率累积 → 熔断 OPEN → `/healthz` 503 |
| 6 | 移除注入 | HALF_OPEN → CLOSED 自愈；`/healthz` 200 |
| 7 | 超限 QPS 压 `/score` | 429 + `ingress.rate_limited` 打点 |
| 8 | 优雅停机 | drain 后 `inflight()==0`、在队请求全部完成 |
| 9 | `GET /metrics` | 分桶直方图/计数/仪表齐全（kv_get/encode_stage2/frontend/circuit/rate_limited） |

**故障注入钩子**：`LocalEmbeddingPS`/`LocalKVStore` 增加 `inject_delay(ms)`（测试专用，生产路径不可达）。

---

## 9. 实施顺序与依赖

```
T0 测试基座 ──▶ T1 真实输入接线 ──▶ T2 可靠性 ──▶ T3 可观测 ──▶ T4 配置/灰度 ──▶ T5 验收
                     │                   ▲                            ▲
                     └───────────────────┴────────────────────────────┘
                        （T2/T3/T4 均以 T1 的最终调用图为观测/保护对象）
```

| 序 | 轨 | 为什么在这个位置 |
|---|---|---|
| 1 | T0 | 先固化「当前正确行为」，此后每步改动有回归网 |
| 2 | T1 | G7 是 P1-High；且 T2 的熔断对象（kv/ps 调用）与 T3 的观测对象（最终调用图）都以接线后的形态为准——先接线避免中间件返工 |
| 3 | T2 | 保护最终调用图（KV get + PS lookup + score） |
| 4 | T3 | 观测 T2 引入的新行为（超时/熔断/背压事件） |
| 5 | T4 | 端点/版本只有接线后才有真实语义 |
| 6 | T5 | 全支柱出口验收 |

- 每轨独立 commit + push（沿用既有提交策略）；每轨完成须 `pytest` 全绿才进下一轨。
- T2 与 T3 在人手充足时可并行（熔断器自带滑动窗口，不依赖 metrics 改造）。

## 10. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R-1 | 前端数值漂移（PS 权重未发布 / 表名错 / dense 拼接顺序错） | 金样本断言（§3.3）+ 发布工具抽样校验（§4.5）+ `ns_groups` 组顺序固化 |
| R-2 | 灰度双版本内存翻倍 | 限 ≤2 版本（§7.4）；Phase 3 C++ 侧权重共享 |
| R-3 | stdlib HTTP 吞吐上限被误当生产入口 | 文档与代码 docstring 明示「验证/冒烟入口」；生产入口 = brpc（Phase 3） |
| R-4 | Python 定时精度（reaper 100ms 粒度） | 语义正确优先：deadline 是上界保证而非精确刻度 |
| R-5 | 分桶化改变 snapshot 语义 | 保留 p50/p99/_count 字段名与插值连续性；`demo.py` 回归验证 |
| R-6 | 左 padding/截尾方向接错（与训练侧 pad 语义冲突） | `encode_s` 显式「截尾保留最新 + 左 padding」并在金样本覆盖「历史超长截断」用例 |

---

## 11. 与既有里程碑编号的对应

| 本阶段轨 | gap_analysis 里程碑 | 差距 |
|---|---|---|
| T0 | （新增，G13 升级 P1） | G13 |
| T1 | M7 前半 | G7 |
| T2 | M6 前半 | G4 |
| T3 | M6 后半 | G5 |
| T4 | M7 后半 | G9 |
| T5 | M6/M7 出口合并 | — |

Phase 3（后续）：M8a/b/c（G6 C++ 移植）、G12（redis/HBM）、G14（局部性能）、OTel 导出器。
