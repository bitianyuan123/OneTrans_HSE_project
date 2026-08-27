# 实现 & 现状总结

> 版本：v1.4（差距评估基线切换：详设目标态 ↔ C++ 工程主线对评，差距编号 D1~D9；正确性以 golden 二进制对拍为准，Python 不作为验证基准）
> 分支：`feat/onetrans-e2e-serving`
> 文档定位（三分体系 ③ 现状 & 差距）：本文记录**已实现/已验证**的落地状态与实测结果；差距分级与路线图见 [gap_analysis.md](./gap_analysis.md)；第二阶段执行设计见 [phase2_design.md](./phase2_design.md)；模型层见 [model_design.md](./model_design.md)；端到端设计见 [detailed_design.md](./detailed_design.md)。

---

## 0. C++ 工程主线（当前形态）

**仓库现状**：`cpp/` 下已存在完整的端到端 C++ 实现，是当前的工程主线。**§7.4 混合架构已落地**：SEDA 分阶段编排（C++ **真实 folly** 线程池）+ Stage II 计算桥（嵌入式 Python 解释器，PyTorch CUDA/CPU 下发算子）+ C++ CPU 数值等价降级路径。Python 单机参照（`onetrans/`、`demo.py`）保留为**一次性 golden 生成源与教学参照**，不再作为在线正确性验证基准。

### 0.1 已交付（C++ 端到端 + 混合计算后端）

| 交付物 | 位置 | 验证 |
|---|---|---|
| `onetrans_core` 静态库 | [cpp/src/](file:///workspace/cpp/src)（common/engine/kv/serving/net） | 数值对拍 24/24 PASS |
| `onetrans_server` HTTP 服务（SEDA + 异步路由） | [cpp/tools/server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) | e2e 8/8 × 3 后端 PASS |
| SEDA 流水线（lookup→encode→kv→batch→score） | [cpp/src/serving/flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) | 并发 32 路 all-200 |
| folly 真实线程池（CPU/IOThreadPoolExecutor 功能分池 + 背压） | [cpp/src/common/executor.cpp](file:///workspace/cpp/src/common/executor.cpp) | `ExecutorOverloaded` 快速失败 |
| Python 计算桥（嵌入解释器 + 专用线程持 GIL） | [cpp/src/serving/compute_bridge.cpp](file:///workspace/cpp/src/serving/compute_bridge.cpp) | golden 对拍 1.49e-07 |
| 桥侧 PyTorch 前向模块 | [cpp/tools/bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) | 与 C++ `score_ns_batch` 等价 |
| `verify_golden` 对拍工具 | [cpp/tools/verify_golden.cpp](file:///workspace/cpp/tools/verify_golden.cpp) | max diff < 1e-6 |
| 权重/golden 导出 | [cpp/tools/export_weights.py](file:///workspace/cpp/tools/export_weights.py) + `gen_golden.py` | manifest + 二进制落盘 |
| 端到端测试（双后端 + 并发断言） | [cpp/tools/e2e_test.py](file:///workspace/cpp/tools/e2e_test.py) | `--backend cpp\|auto\|python` |

### 0.2 实测结果

```
verify_golden：共 24 项，失败 0 项（C++ vs golden.bin 全链路逐位等价，max diff < 1e-6）

e2e_test（8/8 PASS × 3 后端）：
  --backend cpp    ：healthz(backend=cpp) / ingest / score(max 2.38e-07) / kv_miss 全零
                     / concurrent_32(all-200, hit 行对拍 golden) / metrics(flow.backend.cpp) / 404·405
  --backend auto   ：healthz(backend=python, 桥就绪) / score(max 1.49e-07, 经 PyTorch 前向)
                     / concurrent_32 / metrics(flow.backend.python) / 404·405
  --backend python ：同 auto（强制桥路径；桥失败时服务启动即失败，不允许静默降级）

并发：32 路混合 hit/miss score，全 200，0.12s 完成（Python 桥单线程持 GIL 不阻塞接入/编排）
降级：桥队列满（cap=16）或 torch 不可用 → 自动走 C++ `score_ns_batch`（数值等价，e2e 守护）
```

### 0.3 线程模型现状（§7.4 已落地）

**SEDA 阶段链已实现**（[serving/flow.cpp](file:///workspace/cpp/src/serving/flow.cpp)）：每阶段独立线程池 + 有界队列（满则 `ExecutorOverloaded` 快速失败→503），阶段间 `shared_ptr<Ctx>` move 传递所有权；接入层 `route_async` 提交后立即释放 worker，响应由完成线程写回。

| 阶段 | 目标池 | 当前落点 | 状态 |
|---|---|---|---|
| Ingress | brpc bthread（仅收发+序列化） | `net/http_server` accept+worker 池 + `route_async` 异步路由 | **已实现**（HTTP 版；brpc 待换） |
| EmbedLookup | `IOThreadPoolExecutor` | `flow.stage_lookup` @ `embed_lookup` 池 | **已实现** |
| Frontend Encode | `CPUThreadPoolExecutor` | `flow.stage_encode` @ `frontend_encode` 池 | **已实现** |
| KV I/O | `IOThreadPoolExecutor` | `flow.stage_kv` @ `kv_io` 池 | **已实现**（后端 `DatasystemKVStore`/`LocalKVStore` 二选一，经 `KVStore` 接口注入） |
| Stage I Compute | `CPUThreadPoolExecutor` | `nearline` @ `stage1_compute` 池 | **已实现** |
| BatchScheduler | C++ 独立线程 | `flow.batch_loop`（等首条→窗口→满批出批） | **已实现** |
| Python Compute Bridge | 专用线程（持 GIL） | `serving/compute_bridge`（队列满自动降级 cpp） | **已实现** |
| Compute（降级） | `CPUThreadPoolExecutor` | `dispatch_cpp` @ `compute_cpp` 池 | **已实现** |
| 阶段串联 | 回调式串联（`ScoreFlow` 阶段链） | `common/executor`（有界线程池）+ 回调式串联 | **已实现**（回调式） |

---

## 1. 现状总览（C++ 工程主线；正确性以 golden 二进制对拍为准）

序列 Transformer 精排（OneTrans 类）的**单机参照实现**已固化，覆盖「行为流 → 近线 S 侧编码 → UserKV 存储/读取 → 在线 NS 交叉打分」全链路；`demo.py` 仅用于**一次性生成 golden** 与教学演示，**不再作为在线正确性验证口径**。**M5 正确性收口已完成**（G1 元数据固化 / G2 append CAS fencing / G3 路由统一 / G8 KV miss 降级）。正确性基准为 `cpp/artifacts/golden/golden.bin` + `verify_golden`（见 §0）；C++ 主线的接入/数据面/编排见 [detailed_design.md §7.4](./detailed_design.md)。

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

## 5. 工程级差距评估（对照 C++ 主线重评，基线 = 详设目标态）

> 评估口径：**目标=detailed_design.md 的系统设计，现状=cpp/ C++ 工程主线**；Python 侧（`onetrans/`、`cpp/tools/*.py`）分别承担 golden 生成源与验证工具链角色，不作为验证基准。差距编号 D1~D9 的完整分析（差距本质/在线后果/落点设计/验收标准）见 [gap_analysis.md](./gap_analysis.md) 第二部分；此处只列结论。

| 级别 | 差距（对照详设章节） | C++ 现状证据 | 编号 |
|---|---|---|---|
| P1 | 接入层未换装 brpc/bthread（自研 HTTP，无 keep-alive/方法级限流）（§7.4.2 Ingress） | [net/http_server.cpp](file:///workspace/cpp/src/net/http_server.cpp)（`Connection: close`；`route_async` 异步语义已对齐） | D1 |
| P1 | 稀疏 PS 数据面未接入在线热路径：查表为进程内整表直查，无 C++ PS 客户端（§7.4.2/§7.9） | [serving/embed_lookup.h](file:///workspace/cpp/src/serving/embed_lookup.h)（`LookupFn` 替换缝已预留）；`deploy/ps` 存在但 bazel 构建、未接 `cpp/CMakeLists.txt` | D2 |
| P1 | datasystem KV 后端代码就绪但未联调：SDK 未构建部署，运行时实际为 LocalKVStore（§7.4.6） | [kv/datasystem_store.cpp](file:///workspace/cpp/src/kv/datasystem_store.cpp)（对真实 SDK 头校验通过；`ONETRANS_WITH_DATASYSTEM` 默认 OFF） | D3 |
| P1 | 可靠性：无超时/重试/熔断，`/healthz` 不探依赖（§7.8） | [serving/flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) 阶段闭包与 `KVStore::get` 均无 deadline；仅队列背压 | D4 |
| P1 | 可观测性：指标仅 counter/gauge 均值型，无 P99/req_id/trace/Prometheus 格式（§7.8） | [serving/pipeline.h](file:///workspace/cpp/src/serving/pipeline.h) `Metrics` | D5 |
| P1-Mid | 无服务发现/版本注册/灰度（§7.8 版本化） | [server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) CLI 参数 | D6 |
| P1-Mid | Nearline/Online 未分离部署，单进程同宿（§7.9） | [server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) 同挂 `/ingest`+`/score` | D7 |
| P2 | 无 CI 流水线/性能基准回归 | `verify_golden`+`e2e_test` 手工触发 | D8 |
| P2 | 混合参数化层未做 vLLM 自定义 op（M8c） | [tools/bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) eager 前向 | D9 |

> M5 时期的 Python 参照实现差距评估（G1~G14，落点=`onetrans/serving/*.py`）已随基线切换归档：其已修复项（元数据固化/append CAS/路由统一/miss 降级/PS 哈希统一等）由 C++ 主线重新实现并经 golden 对拍承接，逐条处置映射见 [gap_analysis.md](./gap_analysis.md) 第三部分。

---

## 6. 提交策略

- **粒度**：按「修改点 / 功能」独立提交，commit message 用中文「类型: 描述」前缀（`fix:` / `feat:` / `perf:` / `docs:`）。
- **频率**：每个功能点/修复点完成即提交并推送到远端 `origin`（用户约定：代码与文档每次修改后直接 commit + push）。
- **文档**：`docs/` 变更随对应功能同批或独立提交。