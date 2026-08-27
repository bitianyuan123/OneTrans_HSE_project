// OneTrans 精排服务入口：接入层 HTTP + SEDA 编排层 + 数据面 + 混合引擎装配。
//
// 启动：
//   ./onetrans_server --weights cpp/artifacts/weights --port 8080 [--compute-backend auto]
//
// 接口：
//   GET  /healthz                     存活探测（含实际生效计算后端）
//   POST /ingest                      Stage I：行为序列 → UserKV（nearline，全 C++ CPU）
//   POST /score                       Stage II：候选交叉打分（SEDA 流水线，异步路由）
//   GET  /metrics                     Prometheus 风格指标
//
// 计算后端（§7.4.1 混合架构）：
//   python  Stage II 前向经 PythonComputeBridge 下发 PyTorch 算子（CUDA/CPU）
//   cpp     Stage II 前向走 C++ CPU score_ns_batch（数值等价的降级路径）
//   auto    优先 python（torch 可用即启用），失败/队列满自动降级 cpp
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "common/json.h"
#include "common/tensor.h"
#include "engine/frontend.h"
#include "engine/model.h"
#include "engine/two_stage.h"
#include "kv/router.h"
#include "kv/store.h"
#ifdef ONETRANS_WITH_DATASYSTEM
#include "kv/datasystem_store.h"
#endif
#include "net/http_server.h"
#include "serving/compute_bridge.h"
#include "serving/embed_lookup.h"
#include "serving/flow.h"
#include "serving/json_io.h"
#include "serving/pipeline.h"

using namespace onetrans;

namespace {

struct ServerConfig {
    std::string weights_dir = "cpp/artifacts/weights";
    std::string host = "0.0.0.0";
    int port = 8080;
    int http_threads = 8;
    // SEDA 池（§7.4.2）
    int lookup_threads = 4;
    int encode_threads = 2;
    int kv_threads = 4;
    int compute_threads = 0;  // ≤0 → 硬件核数
    size_t queue_cap = 1024;
    size_t max_batch = 32;
    int max_wait_ms = 5;
    int nearline_threads = 2;  // Stage I Compute Pool（nearline prefill）
    std::string model_version = "v42";
    int shards = 1;
    int64_t kv_ttl_seconds = 0;
    std::string kv_backend = "local";       // local | datasystem
    std::string datasystem_host = "127.0.0.1";
    int datasystem_port = 9088;
    std::string compute_backend = "auto";  // auto | python | cpp
    std::string bridge_module_dir = "cpp/tools";
    int bridge_init_timeout_s = 120;
};

ServerConfig parse_args(int argc, char** argv) {
    ServerConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("参数缺值: " + a);
            return argv[++i];
        };
        if (a == "--weights") c.weights_dir = next();
        else if (a == "--host") c.host = next();
        else if (a == "--port") c.port = std::atoi(next().c_str());
        else if (a == "--http-threads") c.http_threads = std::atoi(next().c_str());
        else if (a == "--lookup-threads") c.lookup_threads = std::atoi(next().c_str());
        else if (a == "--encode-threads") c.encode_threads = std::atoi(next().c_str());
        else if (a == "--kv-threads") c.kv_threads = std::atoi(next().c_str());
        else if (a == "--compute-threads") c.compute_threads = std::atoi(next().c_str());
        else if (a == "--nearline-threads") c.nearline_threads = std::atoi(next().c_str());
        else if (a == "--queue-cap") c.queue_cap = static_cast<size_t>(std::atoi(next().c_str()));
        else if (a == "--max-batch") c.max_batch = static_cast<size_t>(std::atoi(next().c_str()));
        else if (a == "--max-wait-ms") c.max_wait_ms = std::atoi(next().c_str());
        else if (a == "--model-version") c.model_version = next();
        else if (a == "--shards") c.shards = std::atoi(next().c_str());
        else if (a == "--kv-ttl-seconds") c.kv_ttl_seconds = std::atoll(next().c_str());
        else if (a == "--kv-backend") c.kv_backend = next();
        else if (a == "--datasystem-host") c.datasystem_host = next();
        else if (a == "--datasystem-port") c.datasystem_port = std::atoi(next().c_str());
        else if (a == "--compute-backend") c.compute_backend = next();
        else if (a == "--bridge-module-dir") c.bridge_module_dir = next();
        else if (a == "--bridge-init-timeout") c.bridge_init_timeout_s = std::atoi(next().c_str());
        else throw std::runtime_error("未知参数: " + a);
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    ServerConfig conf;
    try {
        conf = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "参数错误: " << e.what() << "\n";
        return 2;
    }

    try {
        // ---- 装配：权重 → 引擎 → 数据面 → 桥 → SEDA 编排 → 接入 ---- //
        auto store = std::make_shared<ArtifactStore>();
        store->load(conf.weights_dir, "weights.bin");
        auto model = std::make_shared<OneTransModel>(OneTransModel::load(*store));
        auto frontend = std::make_shared<EmbeddingFrontend>(EmbeddingFrontend::load(*store));
        auto tables = std::make_shared<EmbeddingTables>(EmbeddingTables::load(*store));
        // KV 后端选择（§1.4/§1.5）：datasystem（C++ SDK）或进程内 local 回退。
        std::shared_ptr<KVStore> kv_store;
#ifdef ONETRANS_WITH_DATASYSTEM
        if (conf.kv_backend == "datasystem") {
            DatasystemKVStore::Options ds_opt;
            ds_opt.host = conf.datasystem_host;
            ds_opt.port = conf.datasystem_port;
            ds_opt.default_ttl_seconds = conf.kv_ttl_seconds;
            kv_store = std::make_shared<DatasystemKVStore>(ds_opt);
        }
#endif
        if (!kv_store) {
            kv_store = std::make_shared<LocalKVStore>(conf.kv_ttl_seconds);
        }
        auto metrics = std::make_shared<Metrics>();
        auto runner = std::make_shared<TwoStageRunner>(*model);
        Router router(conf.shards);

        // Python 计算桥（Stage II 算子下发；cpp 后端时跳过启动）
        auto bridge = std::make_shared<PythonComputeBridge>();
        bool want_python = conf.compute_backend == "python" || conf.compute_backend == "auto";
        if (want_python) {
            if (bridge->start(conf.bridge_module_dir, conf.weights_dir,
                              conf.bridge_init_timeout_s)) {
                std::cout << "[onetrans] python bridge ready (Stage II → PyTorch)\n";
            } else if (conf.compute_backend == "python") {
                std::cerr << "[onetrans] python bridge 启动失败: " << bridge->last_error()
                          << "\n";
                return 1;
            } else {
                std::cerr << "[onetrans] python bridge 不可用（降级 C++ CPU）: "
                          << bridge->last_error() << "\n";
            }
        }

        // SEDA 流水线（§7.4.3：lookup → encode → kv → batch → python/cpp 计算）
        ScoreFlow::Config fc;
        fc.lookup_threads = conf.lookup_threads;
        fc.encode_threads = conf.encode_threads;
        fc.kv_threads = conf.kv_threads;
        fc.compute_threads = conf.compute_threads;
        fc.queue_cap = conf.queue_cap;
        fc.max_batch = conf.max_batch;
        fc.batch_wait_ms = conf.max_wait_ms;
        ScoreFlow flow(*frontend, *runner, *kv_store, tables->lookup_fn(), *metrics,
                      want_python ? bridge.get() : nullptr, conf.model_version, fc);
        flow.start();

        // Nearline（§7.4.4：全 C++ CPU，Stage I Compute Pool 异步执行）
        ExecutorSet nearline_execs;
        auto nearline_pool =
            nearline_execs.make("stage1_compute", conf.nearline_threads, conf.queue_cap, false);
        NearlineWorker nearline(*frontend, *runner, *kv_store, *metrics, tables->lookup_fn(),
                                conf.model_version);

        HttpServer server(conf.host, conf.port, conf.http_threads);

        server.route("GET", "/healthz", [&](const HttpRequest&) {
            JsonValue j = JsonValue::object();
            j.set("status", JsonValue::str("ok"));
            j.set("model_version", JsonValue::str(conf.model_version));
            j.set("compute_backend", JsonValue::str(flow.backend()));
            j.set("kv_objects", JsonValue::number(static_cast<double>(kv_store->size())));
            return HttpResponse::json(200, j.dump());
        });

        // 异步路由：接入线程提交后立即释放（响应由完成线程写回，§7.4.3）
        server.route_async("POST", "/ingest", [&](const HttpRequest& req, auto done) {
            try {
                IngestInput in = parse_ingest_input(req.body);
                int shard = router.route(in.user_id);
                metrics->count("ingest.qps");
                nearline_pool->add([&nearline, in = std::move(in), shard, done]() mutable {
                    IngestResult res = nearline.ingest(in);
                    JsonValue j = JsonValue::object();
                    j.set("accepted", JsonValue::boolean(res.accepted));
                    j.set("shard", JsonValue::number(static_cast<double>(shard)));
                    j.set("checksum", JsonValue::str(res.checksum));
                    j.set("reason", JsonValue::str(res.reason));
                    done(HttpResponse::json(res.accepted ? 200 : 400, j.dump()));
                });
            } catch (const std::exception& e) {
                done(HttpResponse::json(400, std::string("{\"error\":\"") + e.what() + "\"}"));
            }
        });

        server.route_async("POST", "/score", [&](const HttpRequest& req, auto done) {
            try {
                ScoreInput in = parse_score_input(req.body);
                std::string user_id = in.user_id;
                int shard = router.route(user_id);
                // 回调式：流水线完成线程直接驱动响应写回（不占接入线程）
                flow.submit(
                    std::move(in),
                    [done, user_id = std::move(user_id), shard](ScoreOutcome out,
                                                                const std::string& err) mutable {
                        if (!err.empty()) {
                            done(HttpResponse::json(
                                503, std::string("{\"error\":\"") + err + "\"}"));
                            return;
                        }
                        JsonValue j = JsonValue::object();
                        j.set("user_id", JsonValue::str(user_id));
                        j.set("shard", JsonValue::number(static_cast<double>(shard)));
                        j.set("kv_hit", JsonValue::boolean(out.kv_hit));
                        j.set("logits", json_parse(logits_to_json(out.logits)));
                        done(HttpResponse::json(200, j.dump()));
                    });
            } catch (const std::exception& e) {
                done(HttpResponse::json(400, std::string("{\"error\":\"") + e.what() + "\"}"));
            }
        });

        server.route("GET", "/metrics", [&](const HttpRequest&) {
            return HttpResponse::text(200, metrics->dump());
        });

        std::cout << "[onetrans] listening on " << conf.host << ":" << conf.port
                  << " model=" << conf.model_version << " shards=" << conf.shards
                  << " batch<=" << conf.max_batch << " wait=" << conf.max_wait_ms << "ms"
                  << " backend=" << (want_python ? "auto→" : "cpp") << flow.backend() << "\n";
        server.run();
        flow.stop();
        nearline_execs.stop_all();
        bridge->stop();
    } catch (const std::exception& e) {
        std::cerr << "[onetrans] 启动失败: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
