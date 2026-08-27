// SHA-256（FIPS 180-4），用于 KV payload checksum 与稳定 hash64——
// 与 Python 侧 hashlib.sha256 / router.hash64 逐位一致（跨语言契约）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onetrans {

// 返回 32 字节摘要
std::vector<uint8_t> sha256(const void* data, size_t len);
std::vector<uint8_t> sha256(const std::string& data);
// 十六进制小写（对齐 Python hashlib.hexdigest()）
std::string sha256_hex(const void* data, size_t len);
std::string sha256_hex(const std::string& data);

// 稳定 64 位键哈希（sha256 前 8 字节大端），对齐 onetrans/serving/router.py:hash64
uint64_t stable_hash64(const std::string& key);

}  // namespace onetrans
