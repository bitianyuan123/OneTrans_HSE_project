# 第二阶段：C++ 端到端精排系统（已落地）

> 版本：v2.0（重写）
> 文档类别：**③ 现状 & 差距 · 阶段执行设计**
> 上游：[detailed_design.md](./detailed_design.md)（目标架构）、[gap_analysis.md](./gap_analysis.md)（差距清单）
> **v1.0 的 Python 单机工程路线已废止**：经评审确认「仓库一行 C++ 都没有、与详细设计差别过大」，Phase 2 重定向为 **C++ 工程主线**，直接落地端到端可用系统。本文描述的即为当前仓库 `cpp/` 下的真实实现。

---

## 0. 阶段定位与产出

Phase 2 的目标不再是「Python 工程形态的验证基准」，而是：

**一条可构建、可运行、可数值对拍的 C++ 端到端链路**——从 HTTP 原始请求进入，经 embedding 查表、两阶段推理（nearline prefill / online 交叉打分）、KV 读写，到 HTTP 返回打分结果，全程无 Python 参与。Python 仅保留两个角色：权重/golden 导出工具，以及被对拍的参照实现。

### 已交付形态

| 交付物 | 位置 | 验证 |
|---|---|---|
| C++17 静态库 `onetrans_core` | `cpp/src/`（五层：common/engine/kv/serving/net） | 单元对拍 24/24 PASS |
| HTTP 精排服务 `onetrans_server` | `cpp/tools/server_main.cpp` | e2e 7/7 PASS |
| 数值对拍工具 `verify_golden` | `cpp/tools/verify_golden.cpp` | max diff < 1e-6 |
| 权重/golden 导出 | `cpp/tools/export_weights.py` + `gen_golden.py` | manifest + 二进制落盘 |
| 端到端测试 | `cpp/tools/e2e_test.py` | ingest→score→对拍 golden |

**依赖约束**：C++ 侧零第三方依赖（仅 C++17 标准库 + pthread），Linux 原生 socket HTTP。构建：`cmake --build build`。

---

## 1. 架构：五层 C++ 组件视图

```
      行为事件(JSON)                          打分请求(JSON, 含候选特征)
            │                                          │
┌───────────▼────────────────────────────────────────── ▼────────────────┐
│ net/http_server          接入层：路由分发 / 请求解析 / 线程池            │
│   POST /ingest  POST /score  GET /healthz  GET /metrics                 │
└───────────┬──────────────────────────────────────────┬─────────────────┘
            │                                          │
┌───────────▼─────────────┐          ┌─────────────────▼─────────────────┐
│ serving/pipeline          │          │ serving/pipeline                  │
│  NearlineWorker.ingest    │          │  Dispatcher（future 化异步提交）   │
│   → embed_lookup 查表     │          │   → BatchScheduler（满批/超时攒批）│
│   → frontend.encode_s     │          │   → OnlineWorker.score_batch       │
│   → runner.encode_s       │          │    → mget(UserKV) + score_ns       │
│   → kv put                │          │    → miss 降级全零 logits           │
└───────┬───────────┬───────┘          └──────────┬───────────┬────────────┘
        │           │                             │           │
┌───────▼──────┐ ┌──▼────────────────┐ ┌─────────▼──────┐ ┌──▼──────────────┐
│ kv/store     │ │ engine/two_stage   │ │ engine/frontend│ │ serving/         │
│ LocalKVStore │ │ encode_s / score_ns│ │ S/NS tokenize  │ │ embed_lookup     │
│ TTL/线程安全 │ │ （Stage I/II）      │ │ pos/RMSNorm    │ │ 四表 mean-bag    │
└──────────────┘ └────────────────────┘ └────────────────┘ └─────────────────┘
        │                    │                                        │
┌───────▼──────┐ ┌──────────▼─────────┐                ┌─────────────▼────────┐
│ kv/serialize │ │ engine/model        │                │ kv/router            │
│ UserKV 二进制 │ │ OneTrans backbone   │                │ Jump/Ring 一致性哈希  │
│ 跨语言兼容    │ │ mixed attn + pyramid│                │ （分片预留）          │
└──────────────┘ └────────────────────┘                └──────────────────────┘
横切：common/（tensor 数值原语 · json · sha256）· serving/pipeline::Metrics（Prometheus 文本）
```

### 分层职责

| 层 | 模块 | 职责 |
|---|---|---|
| common | [tensor.h](../cpp/src/common/tensor.h) / json / sha256 | Tensor 容器、matmul_nt/rms_norm/gelu 原语、手写 JSON、checksum |
| engine | [model.h](../cpp/src/engine/model.h) | OneTrans backbone 单前向（mixed causal self-attention + pyramid 降层 + head） |
| engine | [two_stage.h](../cpp/src/engine/two_stage.h) | Stage I `encode_s`（逐层缓存 K_s/V_s）/ Stage II `score_ns`（读缓存交叉打分）/ `score_ns_batch`（B 候选打包） |
| engine | [frontend.h](../cpp/src/engine/frontend.h) | EmbeddingFrontend：S 侧（item 查表 + mlp + pos + RMSNorm）/ NS 侧（dense + uid/item/artist/album 五组） |
| kv | [serialize.h](../cpp/src/kv/serialize.h) | UserKV 二进制序列化（与 Python 侧逐位兼容，跨语言 golden 对拍） |
| kv | [store.h](../cpp/src/kv/store.h) | LocalKVStore：TTL、mget、sweep、线程安全（生产替换为远端 KV 的协议基线） |
| kv | [router.h](../cpp/src/kv/router.h) | JumpConsistentHash + RingHash（vnode），user→shard 稳定路由 |
| serving | [embed_lookup.h](../cpp/src/serving/embed_lookup.h) | EmbeddingTables 四表加载、LookupFn 回调、multivalent mean-bag |
| serving | [json_io.h](../cpp/src/serving/json_io.h) | HTTP 请求 JSON → IngestInput/ScoreInput；响应装配 |
| serving | [pipeline.h](../cpp/src/serving/pipeline.h) | NearlineWorker / OnlineWorker / BatchScheduler / Dispatcher / Metrics |
| net | [http_server.h](../cpp/src/net/http_server.h) | 极简 HTTP/1.1：路由表、连接线程池、keep-alive |

---

## 2. 数值锚定：Python 参照 → C++ 等价

工程主线的方法论是「数值先行」：**C++ 的每一段计算都有 Python 侧逐位对拍的锚点**，防止移植漂移。

### 2.1 权重导出（`cpp/tools/export_weights.py`）

- 构造与训练侧同构的小配置（D=128, H=4, L=4, max_seq=50, dims=[50,38,27,16,5], Ns=5, ns_group_dims=[240,128,128,128,128]），固定 seed 初始化；
- 一次性导出：backbone 全部权重（blocks 的 norm/W_s/W_ns/final_proj/FFN_s/FFN_ns、head W/b）、embedding 四表（item 1024 / user 256 / artist 128 / album 128）、tokenizer（mlps/type emb/pos emb/RMSNorm）、dense 分箱参数（30 特征 × 8 bins）；
- 落盘 `artifacts/weights/weights.bin`（按名字典索引的连续 float32 blob）+ `manifest.json`（配置 + 每张量名字/shape/offset）。

### 2.2 Golden 生成（`cpp/tools/gen_golden.py`）

- 用同一权重跑 Python 参照实现（`onetrans/serving/` 单机链路），录制中间量与最终量：
  - `s_emb`/`s_mask`（前端 S 侧输出）、`ns_emb`（前端 NS 侧输出）；
  - 逐层 `k_l`/`v_l`（Stage I 缓存，l=0..3）；
  - `logits_two_stage`（两阶段路径）/ `logits_single`（单前向路径）+ `dbg_ns_l`（逐层 NS 隐态，排障用）；
- 输入用例固化在 `artifacts/golden/cases/`（`ingest_case.json` / `score_case.json`），任何人可用相同输入复现。

### 2.3 对拍结果

`verify_golden`（24 项，容差分级 1e-4~5e-4，实测全部 < 1e-6）：

| 组 | 覆盖 |
|---|---|
| frontend.encode_s/s_emb、encode_ns/ns_emb | 前端查表 + tokenize 逐位等价 |
| encode_s/k_0..3、v_0..3 | Stage I 逐层 K/V 缓存等价 |
| score_ns/logits_two_stage | Stage II 交叉打分等价 |
| forward/logits_single | C++ 单前向 vs Python 单前向等价 |
| two_stage vs single 自洽 | C++ 内部两路径互证 |
| kv_serialize/roundtrip/concat | 二进制序列化与 Python 逐位兼容 |
| kvstore/router | LocalKVStore put/get/del、Jump/Ring 哈希行为 |

---

## 3. 服务形态：HTTP 端到端

### 3.1 端点

| 路由 | 语义 |
|---|---|
| `POST /ingest` | `{"user_id", "item_ids", "timestamps"}` → 查表 → encode_s → UserKV put；返回 `{accepted, checksum, s_len}` |
| `POST /score` | `{"user_id", "uid_sparse", "dense", "candidates":[{item_id, artist_ids, album_ids, dense}]}` → mget KV → score_ns；返回 `{logits, kv_hit}`；KV miss 降级全零 logits |
| `GET /healthz` | 存活探针 |
| `GET /metrics` | Prometheus 文本（kv.hit/miss、online.qps、batch、时延累计） |

### 3.2 并发模型

- HTTP 接入线程池（`--http-threads`）解析请求 → `Dispatcher.submit`（future）；
- `BatchScheduler` 满批（`--max-batch`）或超时（`--max-wait-ms`）触发一次 `score_ns_batch`：**一次 mget + 一次批量前向**，miss 请求行降级、顺序保持；
- 打分线程池（`--score-threads`）消费批次，promise 结算；
- nearline ingest 同步执行（写路径，幂等 put 覆盖）。

### 3.3 e2e 验收（`cpp/tools/e2e_test.py`，7/7 PASS）

| 检查 | 断言 |
|---|---|
| healthz | 200 + `status=ok` |
| ingest/accepted | 200 + accepted + s_len=37 |
| score/logits_vs_golden | **max diff 2.38e-07 vs Python golden** |
| score/kv_miss_degrade | 未 ingest 的 user → kv_hit=false + 全零 |
| metrics | kv.hit / online.qps 等指标存在 |
| route/404、route/405 | 未知路径/方法正确拒绝 |

---

## 4. 构建与运行

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 数值对拍（CTest 也注册了同一命令）
./build/verify_golden --weights artifacts/weights --golden artifacts/golden
ctest --test-dir build

# 启动服务
./build/onetrans_server --weights artifacts/weights --port 8080

# 端到端
python3 tools/e2e_test.py --server build/onetrans_server --weights artifacts/weights
```

工件再生成（结构变化时）：`python3 tools/export_weights.py && python3 tools/gen_golden.py`（显式执行并 review diff，防 golden 跟着错误实现漂移）。

---

## 5. 与详细设计的对齐（收敛的差距）

| 差距 | 状态 | 落点 |
|---|---|---|
| G6 C++ 热路径 | **已落地** | `cpp/src/engine/`（model/two_stage/frontend 全部 C++） |
| G7 热路径接线 | **已落地** | `embed_lookup` 四表查表 + frontend 消费原始 JSON 特征 |
| G1/G2 KV 协议 | **已落地** | `kv/serialize` 二进制兼容 + `kv/store` TTL/mget |
| G3 路由 | **已落地** | `kv/router` Jump + Ring（哈希与 Python 跨语言一致） |
| G8 miss 降级 | **已落地** | OnlineWorker miss → 全零 logits，混批顺序保持 |
| G13 回归 | **已落地** | verify_golden（CTest）+ e2e_test 双层 |

**仍在 Phase 3 的**：G4 熔断/重试/限流组件化、G5 直方图分桶 + OTel、G9 配置中心/灰度双版本、G12 远端 KV（redis/HBM 直通）、G14 SIMD/GPU 算子加速、brpc 生产接入。这些的**协议与语义基座**（Metrics 文本、LocalKVStore 接口、Router、UserKV 二进制格式）已在 C++ 侧定型，Phase 3 是替换实现而非重新设计。

---

## 6. Phase 3 展望（基于现有 C++ 基座的增量）

1. **接入层生产化**：`net/http_server` → brpc Server（bthread-per-RPC），路由/线程模型语义不变；
2. **数据面外置**：`kv/store` LocalKVStore → 远端 KV adapter（同一 `put/get/mget` 协议）；`embed_lookup` → PS client（LookupFn 已是回调，注入即换）；
3. **韧性四件套**：超时/重试/熔断/限流，挂在 Dispatcher 与 adapter 调用点（Metrics 打点口已就绪）；
4. **性能**：tensor 原语换 SIMD/Eigen、pyramid KV 显存布局、批量 matmul；以 `verify_golden` + `e2e_test` 守护不回归；
5. **灰度**：双版本 runner 并存（KVKey 已含 model_version），权重共享加载。

---

## 7. 里程碑对应

| 本阶段交付 | gap_analysis 里程碑 |
|---|---|
| cpp/src 引擎层 + 数值对拍 | M8a（C++ 热路径移植）核心 |
| cpp/src serving/kv/net + onetrans_server | M8b（服务编排 C++ 化）+ M6/M7 的 C++ 重述 |
| e2e_test 端到端验收 | M8c（端到端可用）入口验收 |

Phase 1 的 Python 单机参照（`onetrans/`、`demo.py`）保留为**对拍基准与教学参照**，不再承接工程主线；后续所有工程演进默认发生在 `cpp/`。
