// KV 存储：统一逻辑接口 + LocalKVStore（进程内后端）。
//
// 与 onetrans/serving/kv_store.py 的逻辑契约对齐：
// - 键规范 kv:{b64url(model_version)}:{b64url(user_id)}（字符集受限，§1.6）；
// - 记录 = 序列化 payload + 有效长度元数据 + checksum（sha256）；
// - TTL 惰性过期（get 时校验）+ 主动 sweep；
// - LocalKVStore 线程安全（读写锁），供单机部署/单测/小规模生产使用。
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace onetrans {

struct KVKey {
    std::string model_version;
    std::string user_id;

    // datasystem KV key 规范（url-safe base64 去 padding，对齐 Python KVKey.__str__）
    std::string str() const;
    bool operator<(const KVKey& o) const { return str() < o.str(); }
};

struct UserKVRecord {
    KVKey key;
    int64_t s_len = 0;
    std::vector<int64_t> per_layer_len;
    std::string dtype = "float32";
    std::string payload;
    int64_t seq_ts_last = 0;
    int64_t created_at = 0;  // unix 秒

    std::string checksum() const;  // sha256 hex
};

class LocalKVStore {
public:
    explicit LocalKVStore(int64_t default_ttl_seconds = 0) : default_ttl_(default_ttl_seconds) {}

    // 全量写入（幂等覆盖）
    bool put(const UserKVRecord& rec);

    // TTL 过期返回 nullopt（惰性淘汰）
    std::shared_ptr<const UserKVRecord> get(const KVKey& key);

    // 批量读（保持顺序；miss 位置为 nullptr）
    std::vector<std::shared_ptr<const UserKVRecord>> mget(const std::vector<KVKey>& keys);

    int del(const std::vector<KVKey>& keys);

    // 对已有键设置 TTL（expire_at 为 unix 秒；seconds<=0 表示永不过期）
    void ttl(const KVKey& key, int64_t seconds);

    size_t size();

    // 主动回收过期键，返回回收数
    int sweep();

private:
    struct Entry {
        std::shared_ptr<const UserKVRecord> rec;
        int64_t expire_at = 0;  // 0 = 永不过期
    };

    mutable std::mutex mu_;
    std::map<std::string, Entry> map_;
    int64_t default_ttl_;
};

}  // namespace onetrans
