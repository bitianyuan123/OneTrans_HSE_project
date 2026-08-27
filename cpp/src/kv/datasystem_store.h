// Datasystem 后端（yuanrong-datasystem C++ SDK）——实现 KVStore 逻辑接口。
//
// 通过 datasystem::KVClient 直接把 UserKV payload 落到分布式 KV（set/get），
// 与 onetrans/serving/datasystem_adapter.py 的存储语义一致：
//   - key = KVKey::str()（kv:{b64url(mv)}:{b64url(uid)}，字符集受限）；
//   - value = UserKVRecord.payload（kv_serialize 二进制，自含 s_len/per_layer_len 元数据）；
//   - TTL 经 SetParam.ttlSecond / Expire 落地，由 datasystem 服务端回收。
//
// 仅在链接了 datasystem C++ SDK（CMake: ONETRANS_DATASYSTEM=ON 且 find_package(Datasystem)）
// 时才编译本实现；SDK 缺失时热路径回退 LocalKVStore（见 kv/store.h）。
#pragma once

#include "kv/store.h"

#include <memory>

namespace datasystem {
class KVClient;
}

namespace onetrans {

class DatasystemKVStore : public KVStore {
public:
    struct Options {
        std::string host = "127.0.0.1";
        int port = 9088;              // datasystem worker 默认端口（部署时覆盖）
        int64_t default_ttl_seconds = 0;  // 0 = 永不过期
    };

    explicit DatasystemKVStore(const Options& options);
    ~DatasystemKVStore() override;

    bool put(const UserKVRecord& rec) override;
    std::shared_ptr<const UserKVRecord> get(const KVKey& key) override;
    std::vector<std::shared_ptr<const UserKVRecord>> mget(
        const std::vector<KVKey>& keys) override;
    int del(const std::vector<KVKey>& keys) override;
    void ttl(const KVKey& key, int64_t seconds) override;
    size_t size() override { return 0; }  // datasystem 无全量计数；healthz 显示 0
    int sweep() override { return 0; }    // TTL 由服务端惰性回收，客户端无主动 sweep
    const char* backend() const override { return "datasystem"; }

private:
    std::unique_ptr<datasystem::KVClient> client_;
    int64_t default_ttl_;
};

}  // namespace onetrans