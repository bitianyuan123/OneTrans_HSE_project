#include "common/tensor.h"

#include <cmath>
#include <sstream>

#include "common/json.h"
#include "common/sha256.h"

namespace onetrans {

void matmul_nt(const float* x, const float* w, float* y, int64_t n, int64_t in, int64_t out) {
    for (int64_t r = 0; r < n; ++r) {
        const float* xr = x + r * in;
        float* yr = y + r * out;
        for (int64_t o = 0; o < out; ++o) {
            const float* wo = w + o * in;
            float acc = 0.0f;
            for (int64_t i = 0; i < in; ++i) acc += xr[i] * wo[i];
            yr[o] = acc;
        }
    }
}

void rms_norm(const float* x, const float* g, float* out, int64_t rows, int64_t dim, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        float* orow = out + r * dim;
        float ms = 0.0f;
        for (int64_t i = 0; i < dim; ++i) ms += xr[i] * xr[i];
        ms /= static_cast<float>(dim);
        float inv = 1.0f / std::sqrt(ms + eps);
        for (int64_t i = 0; i < dim; ++i) orow[i] = xr[i] * inv * g[i];
    }
}

// --------------------------------------------------------------------------- //
// ArtifactStore
// --------------------------------------------------------------------------- //

static int64_t json_as_int(const JsonValue& v) { return static_cast<int64_t>(v.as_number()); }

void ArtifactStore::load(const std::string& dir, const std::string& bin_file) {
    std::ifstream mf(dir + "/" + "manifest.json");
    if (!mf) throw std::runtime_error("manifest.json 打开失败: " + dir);
    std::stringstream ss;
    ss << mf.rdbuf();
    JsonValue root = json_parse(ss.str());

    const JsonValue& c = root.at("config");
    cfg_.d_model = static_cast<int>(json_as_int(c.at("d_model")));
    cfg_.num_heads = static_cast<int>(json_as_int(c.at("num_heads")));
    cfg_.num_blocks = static_cast<int>(json_as_int(c.at("num_blocks")));
    cfg_.max_seq_len = static_cast<int>(json_as_int(c.at("max_seq_len")));
    cfg_.min_seq_len = static_cast<int>(json_as_int(c.at("min_seq_len")));
    cfg_.ns_tokens_num = static_cast<int>(json_as_int(c.at("ns_tokens_num")));
    cfg_.use_cls_token = c.at("use_cls_token").as_bool();
    for (const auto& v : c.at("dims").as_array()) cfg_.dims.push_back(json_as_int(v));
    cfg_.rms_eps = c.at("rms_eps").as_number();
    cfg_.seq_in_dim = static_cast<int>(json_as_int(c.at("seq_in_dim")));
    for (const auto& v : c.at("ns_group_dims").as_array()) cfg_.ns_group_dims.push_back(json_as_int(v));
    cfg_.num_items = static_cast<int>(json_as_int(c.at("num_items")));
    cfg_.num_users = static_cast<int>(json_as_int(c.at("num_users")));
    cfg_.num_artists = static_cast<int>(json_as_int(c.at("num_artists")));
    cfg_.num_albums = static_cast<int>(json_as_int(c.at("num_albums")));
    cfg_.n_user_dense = static_cast<int>(json_as_int(c.at("n_user_dense")));
    cfg_.n_cand_dense = static_cast<int>(json_as_int(c.at("n_cand_dense")));
    const JsonValue& pw = c.at("piecewise");
    cfg_.piecewise_n_features = static_cast<int>(json_as_int(pw.at("n_features")));
    cfg_.piecewise_max_n_bins = static_cast<int>(json_as_int(pw.at("max_n_bins")));
    for (const auto& v : pw.at("n_bins").as_array()) cfg_.piecewise_n_bins.push_back(json_as_int(v));

    const JsonValue& tensors = root.at("tensors");
    for (const auto& [name, meta] : tensors.as_object()) {
        TensorMeta m;
        m.name = name;
        for (const auto& v : meta.at("shape").as_array()) m.shape.push_back(json_as_int(v));
        m.offset = json_as_int(meta.at("offset"));
        metas_.push_back(std::move(m));
    }

    std::ifstream bf(dir + "/" + bin_file, std::ios::binary);
    if (!bf) throw std::runtime_error(bin_file + " 打开失败: " + dir);
    bf.seekg(0, std::ios::end);
    auto size = static_cast<std::streamoff>(bf.tellg());
    bf.seekg(0, std::ios::beg);
    blob_.resize(static_cast<size_t>(size) / sizeof(float));
    bf.read(reinterpret_cast<char*>(blob_.data()), size);
}

bool ArtifactStore::has(const std::string& name) const {
    for (const auto& m : metas_)
        if (m.name == name) return true;
    return false;
}

const TensorMeta& ArtifactStore::meta(const std::string& name) const {
    for (const auto& m : metas_)
        if (m.name == name) return m;
    throw std::runtime_error("未知张量: " + name);
}

Tensor ArtifactStore::get(const std::string& name) const {
    const TensorMeta& m = meta(name);
    Tensor t(m.shape);
    int64_t n = t.numel();
    std::copy(blob_.begin() + m.offset, blob_.begin() + m.offset + n, t.data.begin());
    return t;
}

}  // namespace onetrans
