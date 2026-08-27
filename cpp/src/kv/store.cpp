#include "kv/store.h"

#include <ctime>

#include "common/sha256.h"

namespace onetrans {

namespace {

// url-safe base64（无 padding），对齐 Python base64.urlsafe_b64encode(...).rstrip("=")
std::string b64url(const std::string& in) {
    static const char kTab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        uint32_t v = (static_cast<uint8_t>(in[i]) << 16) | (static_cast<uint8_t>(in[i + 1]) << 8) |
                     static_cast<uint8_t>(in[i + 2]);
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
        out += kTab[(v >> 6) & 63];
        out += kTab[v & 63];
        i += 3;
    }
    if (i + 1 == in.size()) {
        uint32_t v = static_cast<uint8_t>(in[i]) << 16;
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
    } else if (i + 2 == in.size()) {
        uint32_t v = (static_cast<uint8_t>(in[i]) << 16) | (static_cast<uint8_t>(in[i + 1]) << 8);
        out += kTab[(v >> 18) & 63];
        out += kTab[(v >> 12) & 63];
        out += kTab[(v >> 6) & 63];
    }
    return out;
}

int64_t now_unix() { return static_cast<int64_t>(std::time(nullptr)); }

}  // namespace

std::string KVKey::str() const {
    return "kv:" + b64url(model_version) + ":" + b64url(user_id);
}

std::string UserKVRecord::checksum() const { return sha256_hex(payload); }

bool LocalKVStore::put(const UserKVRecord& rec) {
    Entry e;
    e.rec = std::make_shared<UserKVRecord>(rec);
    e.expire_at = default_ttl_ > 0 ? now_unix() + default_ttl_ : 0;
    std::lock_guard<std::mutex> lk(mu_);
    map_[rec.key.str()] = std::move(e);
    return true;
}

std::shared_ptr<const UserKVRecord> LocalKVStore::get(const KVKey& key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(key.str());
    if (it == map_.end()) return nullptr;
    if (it->second.expire_at > 0 && it->second.expire_at <= now_unix()) {
        map_.erase(it);
        return nullptr;
    }
    return it->second.rec;
}

std::vector<std::shared_ptr<const UserKVRecord>> LocalKVStore::mget(
    const std::vector<KVKey>& keys) {
    std::vector<std::shared_ptr<const UserKVRecord>> out;
    out.reserve(keys.size());
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& k : keys) {
        auto it = map_.find(k.str());
        if (it == map_.end() ||
            (it->second.expire_at > 0 && it->second.expire_at <= now_unix())) {
            out.push_back(nullptr);
        } else {
            out.push_back(it->second.rec);
        }
    }
    return out;
}

int LocalKVStore::del(const std::vector<KVKey>& keys) {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (const auto& k : keys) n += static_cast<int>(map_.erase(k.str()));
    return n;
}

void LocalKVStore::ttl(const KVKey& key, int64_t seconds) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(key.str());
    if (it == map_.end()) return;
    it->second.expire_at = seconds > 0 ? now_unix() + seconds : 0;
}

size_t LocalKVStore::size() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.size();
}

int LocalKVStore::sweep() {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t now = now_unix();
    int n = 0;
    for (auto it = map_.begin(); it != map_.end();) {
        if (it->second.expire_at > 0 && it->second.expire_at <= now) {
            it = map_.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

}  // namespace onetrans
