// UserKV payload 序列化（与 onetrans/serving/serialize.py 二进制兼容）。
//
// 布局：<magic:12B> <header_len:4B LE> <header_json> <raw_bytes>
// raw = concat(K_s^0, V_s^0, ..., K_s^{L-1}, V_s^{L-1})，均 bshd fp32 连续。
// 有效长度元数据（s_len / per_layer_len）固化在 header（G1 契约）。
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "engine/two_stage.h"

namespace onetrans {

struct KvPayload {
    std::vector<int64_t> per_layer_len;
    int64_t s_len = 0;
    int n_layers = 0;
    std::string dtype = "float32";
};

// 序列化（写侧）
std::string kv_serialize(const UserKV& kv);

// 解析 header（不反序列化张量；魔数/布局错误抛异常）
KvPayload kv_read_header(const std::string& payload);

// 反序列化为 UserKV（读侧；per_layer_len/s_len 来自 header）
UserKV kv_deserialize(const std::string& payload);

// append 的数值拼接：base 与 delta 逐层 concat（K、V 各自），返回新 payload。
// 要求两者层数一致且 dtype 一致，否则抛异常。
std::string kv_concat_payload(const std::string& base, const std::string& delta);

}  // namespace onetrans
