# 详细设计 ↔ 工程实现 差距分析

> 版本：v0.5（分析基线切换：`detailed_design.md` 目标设计 ↔ `cpp/` C++ 工程主线 + Python 工具链；正确性以 golden 二进制对拍为唯一口径，Python 运行时不是验证基准）
> 上游：
> - [端到端设计说明书](./e2e_design_spec.md)（概要边界与决策 D1~D6、三 fabric）
> - [详细设计](./detailed_design.md)（KV/Tensor/PS 接口契约、架构/线程模型/部署/可靠性/可观测性全量系统设计）
> 关联：[实现 & 现状总结](./implementation_status.md)（落地状态与实测结果）
> 基线分支：`feat/onetrans-e2e-serving`

---

## 第零部分：本文分析的到底是什么

### 0.1 分析对象与实现基线

**分析对象 = 目标设计 ↔ 当前工程实现**：

- **目标态**：[detailed_design.md](./detailed_design.md) 的系统设计（§6.1 单基准策略、§7.4 线程与并发模型、§7.8 可靠性与可观测性、§7.9 部署视图）；
- **现实态**：仓库当前两个交付面（都是工程实现，角色不同）：

| 交付面 | 位置 | 职责 | 在正确性体系中的角色 |
|---|---|---|---|
| **C++ 工程主线（生产）** | [cpp/src/](file:///workspace/cpp/src) + [cpp/tools/server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) | 接入层 / SEDA 编排 / 数据面 / KV 后端 / Stage I 计算 / Stage II 计算桥 | **被验证对象**：全链路对拍 `golden.bin`（`verify_golden` 24/24） |
| **Python 工具链（工程件）** | [cpp/tools/gen_golden.py](file:///workspace/cpp/tools/gen_golden.py) / `export_weights.py` / [bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) / [e2e_test.py](file:///workspace/cpp/tools/e2e_test.py) | 一次性生成 golden 二进制 / 导出权重 / 桥侧 PyTorch 前向 / e2e 测试驱动 | **验证工具链**：生成基准与测试驱动，不是基准本身 |
| **Python 参照实现（已退役）** | [onetrans/](file:///workspace/onetrans)（含 `serving/*.py`） | 历史参照与教学；`datasystem_adapter.py` 等不在生产热路径 | **golden 生成源**（仅此一职）；不参与在线链路 |

### 0.2 旧 G1~G14 的 Python 落点从何而来

`G1~G14` 及其「落点=`/workspace/onetrans/serving/*.py`」的分析写于 **C++ 主线落地之前**——当时仓库唯一的工程实现是 Python 参照实现，差距自然全部锚定在 `local_adapter.py`、`datasystem_adapter.py`、`dispatcher.py` 等文件上。该分析的**结论**（序列化元数据、CAS、路由统一、miss 降级）已在 C++ 主线重新实现并由 golden 对拍承接，但其**落点与验证方式已过时**。本文按 C++ 主线重评；旧条目逐条处置见第三部分，不再作为正文分析对象。

### 0.3 评估方法

1. 逐条核对「设计承诺 → C++ 实现落点 → 状态」（第一部分映射表）；
2. 未达成项进入第二部分差距详析（新编号 **D1~D9**，全部锚定 C++ 文件与行为）；
3. 每条差距给出：设计要求（详设引用）/ 实现现状（C++ 证据）/ 差距本质 / 在线后果 / 落点设计 / 验收标准；
4. 数值正确性一律以 `verify_golden` 对拍 `golden.bin` 为验收（`max|diff| < 1e-6`），任何差距修复不得以 Python 运行时为验证口径。

---

## 第一部分：设计 → C++ 实现映射总表

> 逐条核对 `detailed_design.md` 承诺与 `cpp/` 现状。状态：✅ 已实现且有验证；⚠️ 部分实现/未联调；❌ 未实现（进入第二部分）。

| # | 设计条目（详设章节） | C++ 实现落点 | 状态 | 差距 |
|---|---|---|---|---|
| 1 | §6.1 golden 二进制单基准（Python 一次性生成，不依赖运行时验证） | [verify_golden.cpp](file:///workspace/cpp/tools/verify_golden.cpp) + `artifacts/golden/` | ✅ 24/24 PASS，max diff < 1e-6 | — |
| 2 | §7.4.1 混合架构：Stage I C++ CPU / Stage II Python CUDA + C++ 降级 | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp)（双后端 dispatch）+ [compute_bridge.cpp](file:///workspace/cpp/src/serving/compute_bridge.cpp) + [two_stage.cpp](file:///workspace/cpp/src/engine/two_stage.cpp) | ✅ `--compute-backend auto\|python\|cpp`，双后端均对拍 golden | — |
| 3 | §7.4.2 功能分池（IO/CPU 分类，有界背压） | [executor.cpp](file:///workspace/cpp/src/common/executor.cpp)：`folly::IOThreadPoolExecutor`（embed_lookup/kv_io）+ `folly::CPUThreadPoolExecutor`（frontend_encode/compute_cpp/stage1_compute）+ `ExecutorOverloaded` 快速失败 | ✅ 真实 folly，`add()` 侧 pending 快照软背压 | 背压为软判定（快照非原子），接入层 429/503 语义已接 |
| 4 | §7.4.3 在线数据流（Lookup→Encode→KV→Batch→Score→回填） | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) `stage_lookup`/`stage_encode`/`stage_kv`/`batch_loop`/`on_batch`/`fill_outcomes` | ✅ 各阶段独立池，`shared_ptr<Ctx>` move 传递 | — |
| 5 | §7.4.4 近线数据流（全程 C++ CPU，不触碰 Python） | [server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) nearline @ `stage1_compute` 池 | ✅ | — |
| 6 | §7.4.5 计算桥（嵌入解释器 + 专用线程持 GIL + 队列满降级） | [compute_bridge.cpp](file:///workspace/cpp/src/serving/compute_bridge.cpp) + [bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) | ✅ cap=16 有界队列；`python` 后端启动失败即失败、`auto` 降级 | — |
| 7 | §7.4.6 KV 存储 datasystem set/get | [datasystem_store.cpp](file:///workspace/cpp/src/kv/datasystem_store.cpp) `DatasystemKVStore`（调 `datasystem::KVClient` Set/Get/Del/Expire） | ⚠️ 代码就绪且按真实 SDK 头文件校验（API 签名一致）；但 SDK 未构建部署，`ONETRANS_WITH_DATASYSTEM` 默认 OFF，**当前运行时实际为 `LocalKVStore`**，未做集群联调 | **D3** |
| 8 | §7.4.7 阶段间协作与背压（回调串联 + 有界 + 乱序回填） | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp)（`ScoreDone` 回调链 + `req` 关联乱序回填） | ✅ | — |
| 9 | §7.4.8 攒批节点（等首条→窗口→满批；miss 前置回填不进计算批） | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) `batch_loop`/`take_batch` | ✅ | — |
| 10 | §7.4.2 Ingress Pool = brpc bthread | [http_server.cpp](file:///workspace/cpp/src/net/http_server.cpp)：自研 HTTP/1.1 线程池 + `route_async` | ❌ 接入层是自研 HTTP（`Connection: close` 无 keep-alive），非 brpc/bthread；无方法级 `max_concurrency` 限流 | **D1** |
| 11 | §7.4.2 EmbedLookup Pool = PS 网络 I/O（远端查表） | [embed_lookup.h](file:///workspace/cpp/src/serving/embed_lookup.h) `EmbeddingTables`（**进程内内存表直查**；`LookupFn` 契约已预留远端替换缝） | ❌ 独立稀疏 PS（[deploy/ps](file:///workspace/deploy/ps)，brpc+proto，bazel 构建）存在但未接入 C++ 热路径，无 C++ PS 客户端 | **D2** |
| 12 | §7.4 GIL 边界（唯一持 GIL 点 = Bridge） | 全编排链路在 C++ folly 池；仅 [compute_bridge.cpp](file:///workspace/cpp/src/serving/compute_bridge.cpp) 线程持 GIL | ✅ e2e 并发 32 路 all-200 验证 | — |
| 13 | §7.8 超时/重试/熔断/健康检查 | 无 deadline 传递、无重试策略、无熔断；`/healthz` 仅自报状态不探依赖 | ❌ | **D4** |
| 14 | §7.8 指标导出（分桶直方图）/结构化日志/trace | [pipeline.h](file:///workspace/cpp/src/serving/pipeline.h) `Metrics`（counter/gauge + Timer 均值型累计）/`/metrics` 文本 | ❌ 无直方图与 P99、无 req_id/trace_id、无 Prometheus 抓取兼容格式 | **D5** |
| 15 | §7.8 版本化（注册中心：版本→checkpoint/表/灰度） | `--model-version` 等 CLI 参数；无注册中心、无灰度开关 | ❌ | **D6** |
| 16 | §7.9 部署视图：Nearline Pool / Online Pool 分离部署 | [server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) 单进程同时挂 `/ingest` 与 `/score` | ❌ 未分离部署（当前单进程同宿） | **D7** |
| 17 | §7.8 优雅停机（摘流量→停收→排空→join） | [http_server.h](file:///workspace/cpp/src/net/http_server.h) `pending_async_` 在途 drain + `Executor::stop()` join 排空 | ⚠️ 基础 drain 有（进程内），无「LB 摘流量→拒绝新请求→排空→退出」全流程 | 并入 **D4** |
| 18 | 回归测试/CI/性能基准 | [e2e_test.py](file:///workspace/cpp/tools/e2e_test.py)（8/8 × 3 后端）+ `verify_golden`（CTest 注册）；无 CI 流水线、无性能基准回归 | ⚠️ | **D8** |
| 19 | （M8c）混合参数化层 vLLM 自定义 op | 桥侧 [bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) PyTorch eager 前向 | ❌ 未做算子融合 | **D9** |

**小结**：详设 §7.4 的**混合架构与数据面主线（#1~#9、#12）已落地且有数值/并发验证**；剩余差距集中在**生产化外围**：接入层形态（D1）、远端数据面（D2/D3）、可靠性（D4）、可观测性（D5）、配置/部署形态（D6/D7）、工程化（D8）与算子融合（D9）。

---

## 第二部分：真实差距详析（D1~D9，全部锚定 C++ 实现）

> 每条按：设计要求 / 实现现状（C++ 证据）/ 差距本质 / 在线后果 / 落点设计 / 验收标准。

### D1：接入层为自研 HTTP，未换装 brpc/bthread（P1）

- **设计要求**（详设 §7.4.2 Ingress Pool、§7.9）：接入层 brpc bthread（M:N 协程），只做「accept + read + JSON 反序列化 + 提交下游 + 收结果写回」，方法级 `max_concurrency` 限流。
- **实现现状**：[http_server.cpp](file:///workspace/cpp/src/net/http_server.cpp) 是自研 HTTP/1.1 线程池模型——`Connection: close`（无 keep-alive，每请求一次 TCP 握手）、worker 线程数固定（默认 8）、`route_async` 已实现「提交即释放 + 完成线程写回」的异步语义（这部分与设计一致）。
- **差距本质**：异步路由**语义**已对齐，缺的是**接入层载体**——无 keep-alive、无 bthread M:N 并发、无方法级并发上限（仅编排层队列背压兜底）、无 brpc 生态（超时/取消/内置监控）。
- **在线后果**：高 QPS 下每请求 TCP 建连开销 + 无连接复用，接入层成为时延与吞吐瓶颈；`deploy/ps` 已用 brpc（可复用其构建与 Controller 超时语义），接入层不统一增加运维面。
- **落点设计**：以 [deploy/ps/embedding_server.cc](file:///workspace/deploy/ps/embedding_server.cc) 为模板新增 `nearline/online` brpc 服务；`route_async` 的 `done` 回调映射为 brpc `Controller`+`Done`；`max_concurrency` 走 brpc 方法选项。`HttpServer` 保留为开发/调试后端。
- **验收标准**：brpc 接入下 `e2e_test`（cpp/python/auto）全绿；keep-alive 压测 QPS 显著优于现 HTTP 版（基准见 D8）。

### D2：稀疏 PS 数据面未接入 C++ 在线热路径（P1）

- **设计要求**（详设 §7.4.2 EmbedLookup Pool、§7.9）：查表是「PS 网络 I/O」，独立稀疏 PS 供 Nearline/Online 共享，支持多表与版本。
- **实现现状**：C++ 热路径查表是 [embed_lookup.h](file:///workspace/cpp/src/serving/embed_lookup.h) 的 `EmbeddingTables`——**进程内整表内存直查**（启动时 `EmbeddingTables::load(store)` 全量加载）。`LookupFn` 契约（[frontend.h L43](file:///workspace/cpp/src/engine/frontend.h#L43)）已显式预留「本地直查 / PS 客户端」替换缝；[deploy/ps](file:///workspace/deploy/ps)（brpc + proto + Knuth 分片 × 多表 × 版本）作为独立服务存在，但用 **bazel** 构建、未纳入 `cpp/CMakeLists.txt`，且**没有 C++ 客户端**（现有 Python 客户端 `embedding_ps_client.py` 不在 C++ 热路径）。
- **差距本质**：整表常驻进程内存——稀疏表超单机内存即不可部署；多副本各持全量表成倍浪费；`flow.stage_lookup`（IO 池）虽已分池，但跑的是内存直查而非网络 I/O，池的类型设定与实际负载不符。
- **在线后果**：模型规模化（大词表 embedding）时单进程内存先爆；PS 与在线服务无法独立扩容；表更新需重启进程（PS 侧已有版本化能力，用不上）。
- **落点设计**：实现 `PsLookupClient` 实现 `LookupFn`（brpc 同步/半同步 `Lookup(table, ids)`，`deploy/ps/embedding_service.proto` 协议已定）；`server_main.cpp` 以 `--embedding-source ps|local` 切换注入；PS 构建纳入统一 CMake 或保留 bazel 但提供产物。
- **验收标准**：`--embedding-source ps` 下 `verify_golden`/`e2e_test` 全绿（数值与 local 查表逐位一致）；PS 挂掉时按设计走确定性 seed 兜底（已有语义）。

### D3：datasystem KV 后端未联调（P1，代码就绪、环境未就绪）

- **设计要求**（详设 §7.4.6）：UserKV 走 datasystem 分布式存储，C++ 经 SDK `KVClient` set/get，TTL 服务端回收。
- **实现现状**：[datasystem_store.cpp](file:///workspace/cpp/src/kv/datasystem_store.cpp) 已按 datasystem 真实 SDK 头（`kv_client.h`/`status.h`/`connection.h`）实现 `Set/Get/Del/Expire`（含 `SetParam.ttlSecond`、`K_NOT_FOUND`→miss 降级、payload 经 `kv_serialize` 自含元数据），并经 `-fsyntax-only` 对 SDK 头校验通过。但：① SDK 需从源码（`yuanrong-datasystem`）`cmake --install` 产出 `DatasystemConfig.cmake` 后 `find_package(Datasystem)` 才命中，当前**未构建部署**；② `ONETRANS_WITH_DATASYSTEM` 默认 OFF，**运行时实际后端是 `LocalKVStore`**；③ 未与真实 datasystem 集群做过联调（`Init` 失败静默降级的路径未被真实触发过）。
- **差距本质**：**实现-部署-验证三层中的后两层缺失**。代码层已收敛到 `KVStore` 接口二选一注入（[server_main.cpp L123-135](file:///workspace/cpp/tools/server_main.cpp#L123)），切换零侵入；缺 SDK 构建产物与集群环境。
- **在线后果**：当前所有「KV 语义」验证（put/get/miss/TTL）都只落在进程内 `LocalKVStore` 上；datasystem 路径的字段映射（如 `created_at` 置 0、`Expire` 取消 TTL 语义）未经真实服务端行为确认。
- **落点设计**：① 部署侧构建 SDK 并 `-DCMAKE_PREFIX_PATH=<sdk>/datasystem/sdk/cpp` 重配 CMake；② `--kv-backend datasystem` 起 worker 集群跑 `e2e_test`；③ 顺带把 `mget` 从逐 key `Get` 换成 SDK 的 `Get(keys[], vals[])` 批量接口（当前实现正确但次优）。
- **验收标准**：datasystem 集群下 `e2e_test --backend cpp` 全绿；`/healthz` 的 `kv_objects` 反映远端规模或显式标注 datasystem 无全量计数；TTL 过期行为实测（写入 `ttl=1s` 后 miss）。

### D4：可靠性四件套缺失——超时/重试/熔断/依赖健康检查（P1）

- **设计要求**（详设 §7.8）：全链路 deadline 传递、读幂等重试、后端错误率熔断、`/healthz` 依赖探针、优雅停机「摘流量→停收→排空→join」。
- **实现现状**：① 各阶段与 KV/查表调用**无 deadline**（[flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) 阶段闭包、`KVStore::get` 均无超时参数；datasystem `Get` 本身支持 `subTimeoutMs` 未透传）；② 无重试（读侧一次失败即 fail_ctx）；③ 无熔断/限流（仅队列满 `ExecutorOverloaded`，这属背压不属熔断）；④ `/healthz` 只自报 `backend/kv_objects`，**不探测 datasystem/PS 依赖**；⑤ 优雅停机有进程内基础（[http_server.h L75](file:///workspace/cpp/src/net/http_server.h#L75) `pending_async_` drain + `Executor::stop()` join 排空），无 LB 摘流量配合。
- **差距本质**：后端（datasystem/PS/桥）任一变慢时无隔离手段——阶段闭包无限等待占住池线程，雪崩沿 SEDA 链向上传导。
- **在线后果**：慢 KV → `kv_io` 池占满 → `ExecutorOverloaded` → 全量 503，且无法区分「过载」与「依赖故障」；发版时在途请求被 join 语义拖住或丢弃。
- **落点设计**：① `KVStore::get` 增 `timeout_ms`（datasystem 透传 `subTimeoutMs`，local 无操作）；② `flow` 各阶段闭包外包一层 deadline（`steady_clock` 截止，超时走 `fail_ctx`）；③ 读侧有限重试（1 次，仅瞬时错误码）；④ `DatasystemKVStore` 暴露 `HealthCheck()`（SDK 自带），`/healthz` 聚合依赖探针；⑤ 停机序列化为「拒新→drain→join」。
- **验收标准**：注入慢后端（datasystem 延迟）时请求在 deadline 内返回错误而非无限挂起；依赖不可达时 `/healthz` 返回 503；停机期在途请求全部完成。

### D5：可观测性——无直方图/P99、无结构化日志、无 trace（P1）

- **设计要求**（详设 §7.8）：分桶直方图 + Prometheus/OTel 导出；req_id/trace_id/user_id 贯穿的结构化日志；分布式 trace 覆盖「路由→读 KV→查表→打分」。
- **实现现状**：[pipeline.h](file:///workspace/cpp/src/serving/pipeline.h) `Metrics` 只有 counter/gauge 两种原语；`Timer` 析构累计 `{name}_us_total` 与 `{name}_n`（**均值型**，无分桶、无 P50/P99）；`/metrics` 输出文本 dump，未按 Prometheus exposition format 校验；全链路**无 req_id/trace_id**（`Ctx` 不携带请求标识）；无任何 trace 埋点。
- **差距本质**：排障三问（哪个请求、哪一段慢、哪个依赖错）一个都答不了；均值型时延掩盖长尾。
- **在线后果**：p99 抖动不可见（均值正常但长尾超标）；并发下无法把慢请求归因到 user/阶段/依赖。
- **落点设计**：① `Metrics` 增加 histogram（指数桶）+ `dump()` 对齐 Prometheus 文本格式；② `ScoreInput`/`Ctx` 增加 `req_id`，入口生成、贯穿阶段回调与 miss/overload 打点；③ trace 以最小 span（lookup/encode/kv/batch/score 五段）先落结构化日志，OTel 后置。
- **验收标准**：`/metrics` 可被 Prometheus 抓取并出 `p99`；一次请求的日志可用 `req_id` 串出五段耗时；`flow.overload_*`/`kv.miss` 事件带 req_id。

### D6：无服务发现 / 模型版本注册（P1-Mid）

- **设计要求**（详设 §7.8 版本化、§7.9 fabric ③）：注册中心维护「model_version → checkpoint 路径 / PS 表版本 / 灰度权重」；端点从发现服务解析而非硬编码。
- **实现现状**：[server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) 全部经 CLI 参数（`--weights/--datasystem-host/--datasystem-port/--model-version`）；datasystem 端点默认 `127.0.0.1:9088`；无灰度开关。
- **在线后果**：切版本=改参数重启；无法双版本并行灰度；实例拓扑变化需人工同步配置。
- **落点设计**：轻量 registry（配置文件或 etcd），`ServerConfig` 启动时 resolve；`model_version` 进 KV key 的既有设计（`kv:{mv}:{uid}`）已为灰度预留键空间。
- **验收标准**：不改代码、仅改 registry 即可完成版本切换与回滚；两版本并存时按 user 哈希分桶路由。

### D7：Nearline/Online 未分离部署（P1-Mid）

- **设计要求**（详设 §7.9）：两池分离部署（写密集 vs 读密集，独立扩容/限流/灰度；UserKV 与 owner 同节点共址）。
- **实现现状**：[server_main.cpp](file:///workspace/cpp/tools/server_main.cpp) 单进程同时提供 `/ingest`（nearline 池）与 `/score`（SEDA 流水线）。进程内功能分离（`stage1_compute` 池与在线各池独立），但部署上同宿。
- **在线后果**：近线批量尾巴（大量 ingest prefill）挤占在线进程的 CPU/内存，无法按负载形态独立扩容；KV 本地性收益（同节点命中）无从谈起。
- **落点设计**：`server_main` 拆 `--role nearline|online|all`（装配层已天然可拆：nearline 只需 frontend+runner+store+lookup，online 只需 flow）；依赖 D1（brpc 服务化）与 D3（KV 外置）先行。
- **验收标准**：两角色独立进程部署，`e2e_test` 走跨进程 ingest→score 全绿；在线进程压测不受近线灌入影响。

### D8：测试/CI/性能基准工程化（P2→P1-Mid）

- **实现现状**：数值守护齐（`verify_golden` 24/24，CTest 已注册；`e2e_test` 8/8 × 3 后端含并发 32 路断言），但无 CI 流水线（提交即手工验证）、无性能基准回归（QPS/p99 无基线追踪）。
- **在线后果**：folly 池参数、攒批窗口等调优无回归护栏；D1/D2 的性能收益无法量化证明。
- **落点设计**：CI（build + verify_golden + e2e_test 三后端）+ 基准脚本（固定负载记 QPS/p99 到 artifacts，趋势对比）。
- **验收标准**：PR 触发自动全绿；性能回归超阈值报警。

### D9：混合参数化层 vLLM 自定义 op（P2，M8c）

- **实现现状**：Stage II 桥侧 [bridge_score.py](file:///workspace/cpp/tools/bridge_score.py) 用 PyTorch eager 实现 `score_ns_batch`（含逐 token 的 `W_ns_list`/`networks_ns_list` 混合参数化路径），数值对拍 golden 达标；未做 GPU 算子融合（vLLM 自定义 op）。
- **在线后果**：GPU 利用率与吞吐有上限（eager 逐层下发 kernel launch 开销）；不影响正确性。
- **落点设计**：详设 M8c 原案不变（混合参数化层落地 vLLM 自定义层/op，复用 KV cache 布局与量化）。
- **验收标准**：op 化后与 `golden.bin` 对拍 `max|diff| < 1e-6` 不回归，吞吐较 eager 有实测提升（D8 基准佐证）。

---

## 第三部分：旧 G1~G14 处置映射（Python 时代 → C++ 时代）

> 旧条目不再展开分析；此处只回答「它去哪了」。

| 旧# | 旧问题（当时落点=Python 参照） | 现处置 | C++ 承接点 / 新编号 |
|---|---|---|---|
| G1 | 序列化丢 `s_len`/`per_layer_len` | ✅ 已承接 | [serialize.cpp](file:///workspace/cpp/src/kv/serialize.cpp) `kv_serialize`/`kv_read_header`（header 自含元数据，golden `kv_roundtrip` 守护） |
| G2 | append「读-合并-写」TOCTOU | ⚠️ 暂不适用 | C++ `KVStore` 接口（[store.h](file:///workspace/cpp/src/kv/store.h)）当前无 append——nearline 为「全量 put 幂等」基线（[pipeline.h L4](file:///workspace/cpp/src/serving/pipeline.h#L4)）；增量 append 引入时需带 CAS（设计保留，不另立编号） |
| G3 | 路由哈希不统一 | ✅ 已承接 | [router.cpp](file:///workspace/cpp/src/kv/router.cpp) jump 哈希为唯一路由源（golden `router/jump_remap_ratio` 守护）；分布式共址待 D7 |
| G4 | 超时/重试/熔断/健康检查 | ↻ 重开 | **D4**（锚定 C++ 阶段闭包与 KVStore 接口） |
| G5 | 指标内存化/无日志/无 trace | ↻ 重开 | **D5**（锚定 C++ `Metrics`） |
| G6 | 无 C++ 热路径 worker | ✅ 主体已承接 | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) + [compute_bridge.cpp](file:///workspace/cpp/src/serving/compute_bridge.cpp)；剩余：**D1**（brpc 接入）、**D9**（vLLM op） |
| G7 | tokenizer+查表未接热路径 | ✅ 已承接 | C++ 侧天然集成：`ScoreInput` 携带原始 id/dense，[frontend.h](file:///workspace/cpp/src/engine/frontend.h) 完成分箱/查表/MLP/RMSNorm 全前处理（`lookup_ns`+`encode_ns_with` 两步式对应 SEDA 阶段） |
| G8 | KV miss 硬失败 | ✅ 已承接 | [flow.cpp](file:///workspace/cpp/src/serving/flow.cpp) miss → 全零 logits + `kv.miss` 打点（e2e `kv_miss` 用例守护） |
| G9 | 无服务发现/注册 | ↻ 重开 | **D6** |
| G10 | PS 跨语言哈希不等价 | ✅ 协议层已收敛 | [embedding_server.cc](file:///workspace/deploy/ps/embedding_server.cc) Knuth 为唯一标准；但 PS 本身未接入热路径 → **D2** |
| G11 | C++ PS 仅单表 | ✅ 已承接 | `deploy/ps` 多表注册 + 版本（同上，接入待 **D2**） |
| G12 | redis 后端 / HBM 直通 | 保留后置 | `KVStore` 接口已为多后端预留（P2） |
| G13 | 无测试框架/CI | ↻ 重开 | **D8** |
| G14 | Python 逐 token 循环等性能 | 大部分消解 | 计算已迁 C++/PyTorch 张量化；残余性能项并入 **D8** 基准护栏 |

---

## 第四部分：分阶段路线图

| 里程碑 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| **R1（数据面收口）** | D3 datasystem SDK 构建 + 集群联调（含 mget 批量化）；D2 C++ PS 客户端接入 `LookupFn` | `--kv-backend datasystem` / `--embedding-source ps` 下 e2e 全绿（golden 对拍不变） | datasystem 部署环境 |
| **R2（可靠性+可观测）** | D4 超时/重试/熔断/依赖探针/停机序列；D5 直方图 + req_id 贯穿 + Prometheus 格式 | 慢后端注入不雪崩；`/metrics` 出 p99；日志可按 req_id 串链 | R1（探针对象存在） |
| **R3（生产形态）** | D1 brpc 接入换装；D7 nearline/online 分离部署；D6 registry/灰度 | 分离部署 e2e 全绿；版本热切换 | R1/R2 |
| **R4（工程化+性能）** | D8 CI + 性能基准；D9 vLLM 自定义 op | CI 全绿门禁；op 化对拍不回归 + 吞吐提升有基准佐证 | R3 |

> 数值守护贯穿所有里程碑：每步以 `verify_golden`（24/24）+ `e2e_test`（三后端）为回归门禁，`golden.bin` 为唯一正确性基准。
