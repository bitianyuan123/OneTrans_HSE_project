#include "kv/serialize.h"

#include <cstring>
#include <stdexcept>

#include "common/json.h"

namespace onetrans {

namespace {

constexpr char kMagic[] = "ONETRANSKV\x01";  // 与 Python _MAGIC 严格一致：11 字节（不含 \0）
constexpr size_t kMagicLen = 11;

struct ParsedHeader {
    JsonValue header;
    size_t body_offset;  // raw bytes 起始偏移
};

size_t tensor_bytes(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return static_cast<size_t>(n) * sizeof(float);
}

ParsedHeader parse_header(const std::string& payload) {
    if (payload.size() < kMagicLen || std::memcmp(payload.data(), kMagic, kMagicLen) != 0)
        throw std::runtime_error("未知 payload 魔数");
    uint32_t hlen;
    std::memcpy(&hlen, payload.data() + kMagicLen, 4);  // 小端
    if (payload.size() < kMagicLen + 4 + hlen)
        throw std::runtime_error("payload header 截断");
    ParsedHeader out;
    out.header = json_parse(payload.substr(kMagicLen + 4, hlen));
    out.body_offset = kMagicLen + 4 + hlen;
    return out;
}

}  // namespace

std::string kv_serialize(const UserKV& kv) {
    JsonValue header = JsonValue::object();
    header.set("dtype", JsonValue::str("float32"));
    header.set("n_layers", JsonValue::number(static_cast<double>(kv.k.size())));
    header.set("s_len", JsonValue::number(static_cast<double>(kv.s_len)));
    JsonValue layers = JsonValue::array({});
    size_t total = 0;
    for (size_t l = 0; l < kv.k.size(); ++l) {
        JsonValue m = JsonValue::object();
        m.set("l", JsonValue::number(static_cast<double>(l)));
        JsonValue ks = JsonValue::array({});
        for (auto s : kv.k[l].shape) ks.push_back(JsonValue::number(static_cast<double>(s)));
        JsonValue vs = JsonValue::array({});
        for (auto s : kv.v[l].shape) vs.push_back(JsonValue::number(static_cast<double>(s)));
        m.set("k_shape", ks);
        m.set("v_shape", vs);
        m.set("len", JsonValue::number(static_cast<double>(kv.per_layer_len[l])));
        layers.push_back(std::move(m));
        total += kv.k[l].data.size() + kv.v[l].data.size();
    }
    header.set("layers", std::move(layers));
    std::string header_json = header.dump();

    std::string payload;
    payload.reserve(kMagicLen + 4 + header_json.size() + total * sizeof(float));
    payload.append(kMagic, kMagicLen);
    uint32_t hlen = static_cast<uint32_t>(header_json.size());
    payload.append(reinterpret_cast<const char*>(&hlen), 4);
    payload.append(header_json);
    for (size_t l = 0; l < kv.k.size(); ++l) {
        payload.append(reinterpret_cast<const char*>(kv.k[l].data.data()),
                       kv.k[l].data.size() * sizeof(float));
        payload.append(reinterpret_cast<const char*>(kv.v[l].data.data()),
                       kv.v[l].data.size() * sizeof(float));
    }
    return payload;
}

KvPayload kv_read_header(const std::string& payload) {
    ParsedHeader p = parse_header(payload);
    KvPayload out;
    out.dtype = p.header.at("dtype").as_string();
    out.s_len = p.header.at("s_len").as_int();
    out.n_layers = static_cast<int>(p.header.at("n_layers").as_int());
    for (const auto& m : p.header.at("layers").as_array())
        out.per_layer_len.push_back(m.at("len").as_int());
    return out;
}

UserKV kv_deserialize(const std::string& payload) {
    ParsedHeader p = parse_header(payload);
    if (p.header.at("dtype").as_string() != "float32")
        throw std::runtime_error("仅支持 float32 payload");

    UserKV kv;
    kv.s_len = p.header.at("s_len").as_int();
    size_t off = p.body_offset;
    for (const auto& m : p.header.at("layers").as_array()) {
        Tensor k({1, 1, 1, 1}), v({1, 1, 1, 1});
        k.shape.clear();
        v.shape.clear();
        for (const auto& s : m.at("k_shape").as_array()) k.shape.push_back(s.as_int());
        for (const auto& s : m.at("v_shape").as_array()) v.shape.push_back(s.as_int());
        k.data.resize(tensor_bytes(k.shape) / sizeof(float));
        v.data.resize(tensor_bytes(v.shape) / sizeof(float));
        if (off + k.data.size() * 4 > payload.size() || off + v.data.size() * 4 > payload.size())
            throw std::runtime_error("payload body 截断");
        std::memcpy(k.data.data(), payload.data() + off, k.data.size() * sizeof(float));
        off += k.data.size() * 4;
        std::memcpy(v.data.data(), payload.data() + off, v.data.size() * sizeof(float));
        off += v.data.size() * 4;
        kv.per_layer_len.push_back(m.at("len").as_int());
        kv.k.push_back(std::move(k));
        kv.v.push_back(std::move(v));
    }
    return kv;
}

std::string kv_concat_payload(const std::string& base, const std::string& delta) {
    ParsedHeader b = parse_header(base);
    ParsedHeader d = parse_header(delta);
    if (b.header.at("n_layers").as_int() != d.header.at("n_layers").as_int())
        throw std::runtime_error("layer_mismatch");

    // 逐层拼接 K/V 数值（S_l += delta_len），重建 header 与 body
    const auto& bl = b.header.at("layers").as_array();
    const auto& dl = d.header.at("layers").as_array();
    JsonValue layers = JsonValue::array({});
    std::string body;
    size_t boff = b.body_offset, doff = d.body_offset;
    for (size_t l = 0; l < bl.size(); ++l) {
        // K
        size_t bkb = tensor_bytes({});
        {
            std::vector<int64_t> shape;
            for (const auto& s : bl[l].at("k_shape").as_array()) shape.push_back(s.as_int());
            bkb = tensor_bytes(shape);
        }
        size_t dkb = 0;
        {
            std::vector<int64_t> shape;
            for (const auto& s : dl[l].at("k_shape").as_array()) shape.push_back(s.as_int());
            dkb = tensor_bytes(shape);
        }
        JsonValue m = JsonValue::object();
        JsonValue ks = JsonValue::array({});
        for (const auto& s : bl[l].at("k_shape").as_array())
            ks.push_back(JsonValue::number(s.as_number() + dl[l].at("k_shape").as_array()[0].as_number()));
        // 仅第 1 维（序列长度）变化
        JsonValue vs = JsonValue::array({});
        for (const auto& s : bl[l].at("v_shape").as_array())
            vs.push_back(JsonValue::number(s.as_number() + dl[l].at("v_shape").as_array()[0].as_number()));
        m.set("l", JsonValue::number(static_cast<double>(l)));
        m.set("k_shape", ks);
        m.set("v_shape", vs);
        m.set("len", JsonValue::number(bl[l].at("len").as_number() + dl[l].at("len").as_number()));
        layers.push_back(std::move(m));

        body.append(base, boff, bkb);
        body.append(delta, doff, dkb);
        boff += bkb;
        doff += dkb;
        // V
        size_t bvb = 0, dvb = 0;
        {
            std::vector<int64_t> shape;
            for (const auto& s : bl[l].at("v_shape").as_array()) shape.push_back(s.as_int());
            bvb = tensor_bytes(shape);
            shape.clear();
            for (const auto& s : dl[l].at("v_shape").as_array()) shape.push_back(s.as_int());
            dvb = tensor_bytes(shape);
        }
        body.append(base, boff, bvb);
        body.append(delta, doff, dvb);
        boff += bvb;
        doff += dvb;
    }

    JsonValue header = JsonValue::object();
    header.set("dtype", JsonValue::str("float32"));
    header.set("n_layers", b.header.at("n_layers"));
    header.set("s_len", JsonValue::number(b.header.at("s_len").as_number() +
                                           d.header.at("s_len").as_number()));
    header.set("layers", std::move(layers));
    std::string header_json = header.dump();

    std::string payload;
    payload.append(kMagic, kMagicLen);
    uint32_t hlen = static_cast<uint32_t>(header_json.size());
    payload.append(reinterpret_cast<const char*>(&hlen), 4);
    payload.append(header_json);
    payload.append(body);
    return payload;
}

}  // namespace onetrans
