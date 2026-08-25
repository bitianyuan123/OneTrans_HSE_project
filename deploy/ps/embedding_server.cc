// 独立稀疏参数服务器（PS）——brpc + bthread（M:N）参考实现。
//
// 设计要点（对应 docs/engineering_design.md「线程模型」与「参数服务」章节）：
// - 稀疏表按 id 稳定哈希切分为 N 个分片，每分片一把细粒度锁（无全局锁，QPS 随核扩展）；
// - brpc::Server 默认 bthread M:N 调度，Lookup 为阻塞式 handler，由 bthread 承载高并发；
// - 未命中返回 0 向量，由客户端按 seed 兜底哈希嵌入重建（权重版本化最差路径）；
// - 表版本（version）随写递增，供在线侧权重版本化加载与失效校验。
//
// 构建：见 BUILD（bazel + brpc/protobuf dep）。

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <brpc/server.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include "embedding_service.pb.h"

DEFINE_int32(port, 8000, "PS 监听端口");
DEFINE_int32(num_shards, 64, "每张表的分片数（分片锁粒度）");
DEFINE_int32(dim, 64, "embedding 维度");
DEFINE_int32(num_bthreads, 0, "brpc bthread worker 数（0=默认，随核）");

namespace onetrans {
namespace ps {

namespace detail {
// 稳定乘法哈希（Knuth），跨进程/版本可复现；与 Python 侧 hash64 语义同构（仅用于分片）。
inline std::size_t ShardOf(int64_t id, int num_shards) {
  constexpr uint64_t kKnuth = 11400714819323198485ULL;
  return static_cast<std::size_t>((static_cast<uint64_t>(id) * kKnuth) % num_shards);
}
}  // namespace detail

// 分片嵌入表：`num_shards` 个独立 HashMap + 每分片锁（细粒度并发）。
class ShardedEmbeddingTable {
 public:
  using Row = std::vector<float>;

  ShardedEmbeddingTable(int num_shards, int dim)
      : num_shards_(num_shards), dim_(dim), version_(0),
        shards_(num_shards), locks_(num_shards) {}

  int dim() const { return dim_; }
  int64_t version() const { return version_.load(std::memory_order_acquire); }

  void Set(int64_t id, Row weights) {
    const std::size_t s = detail::ShardOf(id, num_shards_);
    std::lock_guard<std::mutex> lk(locks_[s]);
    shards_[s].map[id] = std::move(weights);
    version_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 查表：命中写回 out 返回 true；未命中返回 false（调用方填 0）。
  bool Get(int64_t id, Row* out) const {
    const std::size_t s = detail::ShardOf(id, num_shards_);
    std::lock_guard<std::mutex> lk(locks_[s]);
    const auto it = shards_[s].map.find(id);
    if (it == shards_[s].map.end()) return false;
    *out = it->second;
    return true;
  }

 private:
  struct Shard {
    std::unordered_map<int64_t, Row> map;
  };

  int num_shards_;
  int dim_;
  std::atomic<int64_t> version_;
  std::vector<Shard> shards_;
  mutable std::vector<std::mutex> locks_;
};

// RPC 服务实现（brpc：handler 运行于 bthread，天然 M:N）。
class EmbeddingServiceImpl : public EmbeddingService {
 public:
  explicit EmbeddingServiceImpl(ShardedEmbeddingTable* table) : table_(table) {}

  void Lookup(google::protobuf::RpcController* /*cntl*/,
              const EmbeddingLookupRequest* req,
              EmbeddingLookupResponse* resp,
              google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    DoLookup(*req, resp);
  }

  void BatchLookup(google::protobuf::RpcController* /*cntl*/,
                   const EmbeddingBatchRequest* req,
                   EmbeddingBatchResponse* resp,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    for (const auto& sub : req->requests()) {
      DoLookup(sub, resp->add_responses());
    }
  }

 private:
  void DoLookup(const EmbeddingLookupRequest& req, EmbeddingLookupResponse* resp) const {
    const int dim = table_->dim();
    resp->set_table(req.table());
    resp->set_dim(dim);
    resp->set_version(table_->version());
    resp->mutable_weights()->Reserve(req.ids_size() * dim);
    for (const int64_t id : req.ids()) {
      ShardedEmbeddingTable::Row row;
      const bool hit = table_->Get(id, &row);
      resp->add_ids(id);
      for (int d = 0; d < dim; ++d) {
        // 未命中回 0：客户端 / 上层用 seed 兜底哈希嵌入重建（权重版本化最差路径）。
        resp->add_weights(hit ? row[d] : 0.0f);
      }
    }
  }

  ShardedEmbeddingTable* table_;  // 不拥有（由 main 持有）
};

}  // namespace ps
}  // namespace onetrans

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  onetrans::ps::ShardedEmbeddingTable table(FLAGS_num_shards, FLAGS_dim);
  onetrans::ps::EmbeddingServiceImpl impl(&table);

  brpc::Server server;
  if (server.AddService(&impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "AddService 失败";
    return 1;
  }

  brpc::ServerOptions options;
  if (FLAGS_num_bthreads > 0) {
    options.num_threads = FLAGS_num_bthreads;
  }

  if (server.Start(FLAGS_port, &options) != 0) {
    LOG(ERROR) << "PS 启动失败，端口 " << FLAGS_port;
    return 1;
  }

  LOG(INFO) << "embedding PS 监听 :" << FLAGS_port
            << " (shards=" << FLAGS_num_shards << ", dim=" << FLAGS_dim << ")";
  server.RunUntilAskedToQuit();
  return 0;
}