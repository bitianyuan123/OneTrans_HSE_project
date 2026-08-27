#include "kv/router.h"

#include <algorithm>

namespace onetrans {

namespace {
// 环上二分：首个 token > h 的虚拟节点（环形回绕到 0）
const std::pair<uint64_t, std::string>& ring_upper_bound(
    const std::vector<std::pair<uint64_t, std::string>>& ring, uint64_t h) {
    // bisect_right(tokens, h)：首个 token > h
    size_t lo = 0, hi = ring.size();
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (ring[mid].first <= h)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == ring.size()) lo = 0;  // 环回绕
    return ring[lo];
}
}  // namespace

void RingHash::add_node(const std::string& node) {
    if (nodes_.count(node)) return;
    nodes_.insert(node);
    for (int i = 0; i < vnodes_; ++i) {
        uint64_t token = stable_hash64(node + "#" + std::to_string(i));
        auto it = std::lower_bound(ring_.begin(), ring_.end(), token,
                                   [](const auto& p, uint64_t t) { return p.first < t; });
        ring_.insert(it, {token, node});
    }
}

void RingHash::remove_node(const std::string& node) {
    if (!nodes_.count(node)) return;
    nodes_.erase(node);
    ring_.erase(std::remove_if(ring_.begin(), ring_.end(),
                               [&](const auto& p) { return p.second == node; }),
                ring_.end());
}

std::string RingHash::shard_of(const std::string& key) const {
    if (ring_.empty()) throw std::runtime_error("环为空，请先 add_node");
    return ring_upper_bound(ring_, stable_hash64(key)).second;
}

Router::Router(int num_shards) : jump_(num_shards), ring_(1) {}

Router::Router(RingHash ring) : jump_(1), ring_(std::move(ring)), use_ring_(true) {
    // 节点名快照（有序去重），route() 返回节点在快照中的序号作为 owner 索引
    for (const auto& node : ring_.nodes()) ring_nodes_.push_back(node);
}

int Router::route(const std::string& user_id) const {
    if (!use_ring_) return jump_.shard_of(user_id);
    // ring 模式：返回节点在快照中的序号（owner 索引）
    std::string node = ring_.shard_of(user_id);
    auto it = std::find(ring_nodes_.begin(), ring_nodes_.end(), node);
    if (it == ring_nodes_.end()) throw std::runtime_error("路由节点不在快照: " + node);
    return static_cast<int>(it - ring_nodes_.begin());
}

}  // namespace onetrans
