// OneTrans 精排服务入口：接入层 HTTP + 编排层 + 数据面 + 引擎全链路装配。
//
// 启动：
//   ./onetrans_server --weights cpp/artifacts/weights --port 8080
//
// 接口：
//   GET  /healthz                     存活探测
//   POST /ingest                      Stage I：用户行为序列 → User KV（nearline）
//   POST /score                       Stage II：候选交叉打分（online，动态攒批）
//   GET  /metrics                     Prometheus 风格指标
//
// 请求体即 gen_golden.py 的 cases/*.json（数值对拍锚点）。
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
#include "net/http_server.h"
#include "serving/embed_lookup.h"
#include "serving/json_io.h"
#include "serving/pipeline.h"

using namespace onetrans;

namespace {

struct ServerConfig {
    std::string weights_dir = "cpp/artifacts/weights";
    std::string host = "0.0.0.0";
    int port = 8080;
    int http_threads = 8;
    int score_threads = 2;
    size_t max_batch = 32;
    int max_wait_ms = 5;
    std::string model_version = "v42";
    int shards = 1;
    int64_t kv_ttl_seconds = 0;
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
        else if (a == "--score-threads") c.score_threads = std::atoi(next().c_str());
        else if (a == "--max-batch") c.max_batch = static_cast<size_t>(std::atoi(next().c_str()));
        else if (a == "--max-wait-ms") c.max_wait_ms = std::atoi(next().c_str());
        else if (a == "--model-version") c.model_version = next();
        else if (a == "--shards") c.shards = std::atoi(next().c_str());
        else if (a == "--kv-ttl-seconds") c.kv_ttl_seconds = std::atoll(next().c_str());
        else throw std::runtime_error("未知参数: " + a);
    }
    return c;
}

// --------------------------------------------------------------------------- //
// 请求体解析（JSON → 业务输入）：见 serving/json_io
// --------------------------------------------------------------------------- //

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
        // ---- 装配：权重 → 引擎 → 数据面 → 编排 → 接入 ---- //
        auto store = std::make_shared<ArtifactStore>();
        store->load(conf.weights_dir, "weights.bin");
        auto model = std::make_shared<OneTransModel>(OneTransModel::load(*store));
        auto frontend = std::make_shared<EmbeddingFrontend>(EmbeddingFrontend::load(*store));
        auto tables = std::make_shared<EmbeddingTables>(EmbeddingTables::load(*store));
        auto kv_store = std::make_shared<LocalKVStore>(conf.kv_ttl_seconds);
        auto metrics = std::make_shared<Metrics>();
        auto runner = std::make_shared<TwoStageRunner>(*model);
        Router router(conf.shards);

        NearlineWorker nearline(*frontend, *runner, *kv_store, *metrics, tables->lookup_fn(),
                                conf.model_version);
        OnlineWorker online(*frontend, *runner, *kv_store, *metrics, tables->lookup_fn(),
                            conf.model_version);
        BatchScheduler sched(conf.max_batch, std::chrono::milliseconds(conf.max_wait_ms));
        Dispatcher dispatcher(online, sched, conf.score_threads, *metrics);
        dispatcher.start();

        HttpServer server(conf.host, conf.port, conf.http_threads);

        server.route("GET", "/healthz", [&](const HttpRequest&) {
            JsonValue j = JsonValue::object();
            j.set("status", JsonValue::str("ok"));
            j.set("model_version", JsonValue::str(conf.model_version));
            j.set("kv_objects", JsonValue::number(static_cast<double>(kv_store->size())));
            return HttpResponse::json(200, j.dump());
        });

        server.route("POST", "/ingest", [&](const HttpRequest& req) {
            IngestInput in = parse_ingest_input(req.body);
            int shard = router.route(in.user_id);
            metrics->count("ingest.qps");
            IngestResult res = nearline.ingest(in);
            JsonValue j = JsonValue::object();
            j.set("accepted", JsonValue::boolean(res.accepted));
            j.set("shard", JsonValue::number(static_cast<double>(shard)));
            j.set("checksum", JsonValue::str(res.checksum));
            j.set("reason", JsonValue::str(res.reason));
            return HttpResponse::json(res.accepted ? 200 : 400, j.dump());
        });

        server.route("POST", "/score", [&](const HttpRequest& req) {
            ScoreInput in = parse_score_input(req.body);
            const std::string user_id = in.user_id;
            int shard = router.route(user_id);
            auto fut = dispatcher.submit(std::move(in));
            ScoreOutcome out = fut.get();  // 攒批窗口内聚合，时延有界
            JsonValue j = JsonValue::object();
            j.set("user_id", JsonValue::str(user_id));
            j.set("shard", JsonValue::number(static_cast<double>(shard)));
            j.set("kv_hit", JsonValue::boolean(out.kv_hit));
            j.set("logits", json_parse(logits_to_json(out.logits)));
            return HttpResponse::json(200, j.dump());
        });

        server.route("GET", "/metrics", [&](const HttpRequest&) {
            return HttpResponse::text(200, metrics->dump());
        });

        std::cout << "[onetrans] listening on " << conf.host << ":" << conf.port
                  << " model=" << conf.model_version << " shards=" << conf.shards
                  << " batch<=" << conf.max_batch << " wait=" << conf.max_wait_ms << "ms\n";
        server.run();
        dispatcher.stop();
    } catch (const std::exception& e) {
        std::cerr << "[onetrans] 启动失败: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
