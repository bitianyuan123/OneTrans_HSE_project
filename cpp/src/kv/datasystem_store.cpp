#include "kv/datasystem_store.h"

#include <datasystem/kv_client.h>

#include "kv/serialize.h"

namespace onetrans {

namespace {

// 从 datasystem 取回的 payload 重建 UserKVRecord（元数据回读 header）。
std::shared_ptr<const UserKVRecord> build_record(const KVKey& key, std::string payload) {
    if (payload.empty()) return nullptr;
    try {
        auto rec = std::make_shared<UserKVRecord>();
        rec->key = key;
        rec->payload = std::move(payload);
        KvPayload hdr = kv_read_header(rec->payload);
        rec->s_len = hdr.s_len;
        rec->per_layer_len = hdr.per_layer_len;
        rec->dtype = hdr.dtype;
        rec->created_at = 0;  // datasystem 未单独记录，置 0（计算只依赖 payload）
        return rec;
    } catch (const std::exception&) {
        return nullptr;  // 损坏/旧格式 → 按 miss 降级
    }
}

}  // namespace

DatasystemKVStore::DatasystemKVStore(const Options& options)
    : default_ttl_(options.default_ttl_seconds) {
    datasystem::ConnectOptions co;
    co.host = options.host;
    co.port = options.port;
    client_ = std::make_unique<datasystem::KVClient>(co);
    // Init 建立与 worker 的连接（空配置：走环境变量/默认）；失败不抛，
    // 后续 put/get 返回错误并由 G8 miss 语义统一降级。
    (void)client_->Init();
}

DatasystemKVStore::~DatasystemKVStore() {
    if (client_) client_->ShutDown();
}

bool DatasystemKVStore::put(const UserKVRecord& rec) {
    datasystem::SetParam param;
    if (default_ttl_ > 0) param.ttlSecond = static_cast<uint32_t>(default_ttl_);
    datasystem::Status s = client_->Set(rec.key.str(), rec.payload, param);
    return s.IsOk();
}

std::shared_ptr<const UserKVRecord> DatasystemKVStore::get(const KVKey& key) {
    std::string val;
    datasystem::Status s = client_->Get(key.str(), val);
    // 未命中 / 传输错误统一降级为 miss（与 LocalKVStore 的 not-found 语义一致）。
    if (s.GetCode() == datasystem::K_NOT_FOUND || s.IsError()) return nullptr;
    return build_record(key, std::move(val));
}

std::vector<std::shared_ptr<const UserKVRecord>> DatasystemKVStore::mget(
    const std::vector<KVKey>& keys) {
    std::vector<std::shared_ptr<const UserKVRecord>> out(keys.size());
    // SEDA 热路径按请求逐 key get（KV I/O 池并发）；mget 仅用于测试/旧同步路径，
    // 逐 key 保证顺序与 miss 语义严格一致。
    for (size_t i = 0; i < keys.size(); ++i) out[i] = get(keys[i]);
    return out;
}

int DatasystemKVStore::del(const std::vector<KVKey>& keys) {
    if (keys.empty()) return 0;
    std::vector<std::string> kstr;
    kstr.reserve(keys.size());
    for (const auto& k : keys) kstr.push_back(k.str());
    std::vector<std::string> failed;
    datasystem::Status s = client_->Del(kstr, failed);
    if (s.IsError()) return 0;
    return static_cast<int>(keys.size() - failed.size());
}

void DatasystemKVStore::ttl(const KVKey& key, int64_t seconds) {
    std::vector<std::string> failed;
    // seconds<=0 → Expire 传 0，表示取消 TTL（改由显式 Del 管理生命周期）。
    uint32_t ttl = seconds > 0 ? static_cast<uint32_t>(seconds) : 0;
    (void)client_->Expire({key.str()}, ttl, failed);
}

}  // namespace onetrans