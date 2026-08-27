// 一致性哈希路由（user → owner worker 数据本地化）。
//
// 与 onetrans/serving/router.py 逐位对齐（跨语言一致性契约）：
// - JumpConsistentHash：Lamping-Veach 跳变哈希，桶数固定时 remap 最优；
// - RingHash：虚拟节点环（ketama 风格），支持动态增删；
// - 键哈希统一 sha256 前 8 字节大端（stable_hash64），跨进程/重启可复现。
#pragma once

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/sha256.h"

namespace onetrans {

class JumpConsistentHash {
public:
    explicit JumpConsistentHash(int num_shards) : num_shards_(num_shards) {
        if (num_shards <= 0) throw std::invalid_argument("num_shards 必须 ≥ 1");
    }

    // 返回 key 归属分片索引 [0, num_shards)（算法与 Python 版逐位一致）
    int shard_of(const std::string& key) const {
        uint64_t k = stable_hash64(key);
        int64_t b = -1, j = 0;
        while (j < num_shards_) {
            b = j;
            k = k * 2862933555777941757ULL + 1ULL;
            j = static_cast<int64_t>((static_cast<double>(b) + 1.0) * 2147483648.0 /
                                     static_cast<double>((k >> 33) + 1ULL));
        }
        return static_cast<int>(b);
    }

private:
    int64_t num_shards_;
};

class RingHash {
public:
    explicit RingHash(int vnodes_per_node = 128) : vnodes_(vnodes_per_node) {}

    void add_node(const std::string& node);
    void remove_node(const std::string& node);

    // key 归属节点（环为空抛异常）
    std::string shard_of(const std::string& key) const;

    const std::set<std::string>& nodes() const { return nodes_; }
    size_t node_count() const { return nodes_.size(); }

private:
    // (token, node)
    std::vector<std::pair<uint64_t, std::string>> ring_;  // 按 token 升序
    std::set<std::string> nodes_;
    int vnodes_;
};

// 路由门面：user_id → owner shard 索引。
// num_shards 模式内部用 jump hash；ring 模式用虚拟节点环（返回节点序号）。
class Router {
public:
    explicit Router(int num_shards);
    explicit Router(RingHash ring);

    int route(const std::string& user_id) const;

private:
    JumpConsistentHash jump_;
    RingHash ring_;
    bool use_ring_ = false;
    std::vector<std::string> ring_nodes_;  // 环成员快照（索引 ↔ 节点名）
};

}  // namespace onetrans
