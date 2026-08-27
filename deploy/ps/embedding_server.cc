// 独立稀疏参数服务器（PS）——brpc + bthread（M:N）参考实现。
//
// 设计要点（对应 docs/detailed_design.md §4.5「稀疏 PS 数据面契约」与 §7.4「线程与并发模型」）：
// - 稀疏表按 id 稳定哈希切分为 N 个分片，每分片一把细粒度锁（无全局锁，QPS 随核扩展）；
// - 分片路由统一 Knuth 乘法哈希（detail::ShardOf 为唯一标准，Python 侧
//   embedding_ps_client.py 的 ShardedEmbeddingTable.shard_of 与其逐位对齐，G10）；
// - 多表支持：table 名 -> ShardedEmbeddingTable 注册中心（按需懒建 + 淘汰），
//   DoLookup 按 req.table() 路由，支撑多模型版本并存与灰度（G11）；
// - brpc::Server 默认 bthread M:N 调度，Lookup 为阻塞式 handler，由 bthread 承载高并发；
// - 未命中返回 0 向量，由客户端按 seed 兜底哈希嵌入重建（权重版本化最差路径）；
// - 表版本（version）随写递增，供在线侧权重版本化加载与失效校验。
//
// 构建：见 BUILD（bazel + brpc/protobuf dep）。

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
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
// 稳定乘法哈希（Knuth）：**分片路由唯一标准**（G10）。
// Python 侧 ShardedEmbeddingTable.shard_of 用同一常数、同一二补码回绕与 2^64 截断语义，
// 保证读写两侧对同一 id 落同一分片。
inline std::size_t ShardOf(int64_t id, int num_shards) {
  constexpr uint64_t kKnuth = 11400714819323198485ULL;  // 0x9E3779B97F4A7C15
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

// 多表注册中心：table 名 -> ShardedEmbeddingTable（G11）。
//
// 与 Python 侧 LocalEmbeddingPS 的多表语义对齐：
// - 表按需懒建（Get 首次触达即注册），返回 shared_ptr 快照，读路径持锁仅覆盖注册动作；
// - Erase 淘汰整表（模型版本下线/回滚）；淘汰后进行中的旧请求仍持 shared_ptr 安全读；
// - 每表独立 version（写递增），供「多模型版本并存 + 按版本查表 + 权重版本化失效」。
class TableRegistry {
 public:
  using TablePtr = std::shared_ptr<ShardedEmbeddingTable>;

  TableRegistry(int num_shards, int dim) : num_shards_(num_shards), dim_(dim) {}

  // 取表；不存在则注册新表（懒建，对齐 Python LocalEmbeddingPS.table()）。
  TablePtr Get(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = tables_.find(name);
    if (it != tables_.end()) return it->second;
    auto t = std::make_shared<ShardedEmbeddingTable>(num_shards_, dim_);
    tables_.emplace(name, t);
    return t;
  }

  // 只读查找（不注册）：不存在的表返回 false（调用方走 miss/兜底路径）。
  bool Find(const std::string& name, TablePtr* out) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = tables_.find(name);
    if (it == tables_.end()) return false;
    *out = it->second;
    return true;
  }

  // 淘汰整表（版本下线/回滚）；返回是否确实删除。
  bool Erase(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    return tables_.erase(name) > 0;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return tables_.size();
  }

 private:
  const int num_shards_;
  const int dim_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, TablePtr> tables_;
};

// RPC 服务实现（brpc：handler 运行于 bthread，天然 M:N）。
class EmbeddingServiceImpl : public EmbeddingService {
 public:
  explicit EmbeddingServiceImpl(TableRegistry* registry) : registry_(registry) {}

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
    // G11：按 req.table() 路由到对应分片表（多模型版本/灰度）；懒建后未写入即全 miss。
    const TableRegistry::TablePtr table = registry_->Get(req.table());
    const int dim = table->dim();
    resp->set_table(req.table());
    resp->set_dim(dim);
    resp->set_version(table->version());
    resp->mutable_weights()->Reserve(req.ids_size() * dim);
    for (const int64_t id : req.ids()) {
      ShardedEmbeddingTable::Row row;
      const bool hit = table->Get(id, &row);
      resp->add_ids(id);
      for (int d = 0; d < dim; ++d) {
        // 未命中回 0：客户端 / 上层用 seed 兜底哈希嵌入重建（权重版本化最差路径）。
        resp->add_weights(hit ? row[d] : 0.0f);
      }
    }
  }

  TableRegistry* registry_;  // 不拥有（由 main 持有）
};

}  // namespace ps
}  // namespace onetrans

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  onetrans::ps::TableRegistry registry(FLAGS_num_shards, FLAGS_dim);
  onetrans::ps::EmbeddingServiceImpl impl(&registry);

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