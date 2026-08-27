#include "engine/frontend.h"

#include <cmath>
#include <stdexcept>

namespace onetrans {

namespace {

// _mlp：Linear(in→out) → GELU → Linear(out→out)（均无 bias）
struct Mlp {
    Tensor w1, w2;
    void apply(const float* x, float* y, int64_t rows, int64_t in, int64_t out) const {
        std::vector<float> h(static_cast<size_t>(rows * out));
        linear_rows(x, w1.data.data(), h.data(), rows, in, out);
        for (auto& v : h) v = gelu(v);
        linear_rows(h.data(), w2.data.data(), y, rows, out, out);
    }
};

// piecewise 分箱编码（对齐 onetrans/nn/encoders/piecewise.py forward）
// x [M, n_features] → out [M, sum(n_bins)]（行主序：feature 主、bin 次）
Tensor piecewise_forward(const Tensor& w, const Tensor& b, const std::vector<int64_t>& n_bins,
                         const float* x, int64_t M, int n_features) {
    const int64_t max_bins = w.shape[1];
    int64_t out_dim = 0;
    for (auto nb : n_bins) out_dim += nb;
    Tensor out({M, out_dim});
    for (int64_t m = 0; m < M; ++m) {
        const float* xm = x + m * n_features;
        float* om = out.data.data() + m * out_dim;
        int64_t o = 0;
        for (int i = 0; i < n_features; ++i) {
            const int64_t nb = n_bins[i];
            for (int64_t j = 0; j < nb; ++j, ++o) {
                float v = b.at(i * max_bins + j) + w.at(i * max_bins + j) * xm[i];
                if (nb == 1) {
                    // single bin：x >= bias ? 1 : 0
                    v = (xm[i] >= b.at(i * max_bins)) ? 1.0f : 0.0f;
                } else if (j == 0) {
                    v = std::min(v, 1.0f);
                } else if (j == nb - 1) {
                    v = std::max(v, 0.0f);
                } else {
                    v = std::min(std::max(v, 0.0f), 1.0f);
                }
                om[o] = v;
            }
        }
    }
    return out;
}

}  // namespace

EmbeddingFrontend EmbeddingFrontend::load(const ArtifactStore& store) {
    EmbeddingFrontend f;
    f.cfg_ = store.config();
    const int D = f.cfg_.d_model;

    f.s_mlp_w1 = store.get("tokenizer.s_tokenizer.mlps.0.0.weight");
    f.s_mlp_w2 = store.get("tokenizer.s_tokenizer.mlps.0.2.weight");
    f.s_type_emb = store.get("tokenizer.s_tokenizer.type_embeddings.weight");
    f.pos_emb = store.get("tokenizer.pos_embedding.weight");
    f.tok_rms_g = store.get("tokenizer.rms_norm.weight");

    f.ns_w1.resize(f.cfg_.ns_group_dims.size());
    f.ns_w2.resize(f.cfg_.ns_group_dims.size());
    for (size_t g = 0; g < f.cfg_.ns_group_dims.size(); ++g) {
        f.ns_w1[g] = store.get("tokenizer.ns_tokenizer.mlps." + std::to_string(g) + ".0.weight");
        f.ns_w2[g] = store.get("tokenizer.ns_tokenizer.mlps." + std::to_string(g) + ".2.weight");
    }
    f.pw_w = store.get("embedder.piecewise_encoder.weight");
    f.pw_b = store.get("embedder.piecewise_encoder.bias");
    return f;
}

std::pair<Tensor, Tensor> EmbeddingFrontend::encode_s(const std::vector<int64_t>& item_ids,
                                                      const LookupFn& lookup) const {
    const int S = cfg_.max_seq_len;
    const int D = cfg_.d_model;
    const float eps = static_cast<float>(cfg_.rms_eps);

    // 截尾保留最新 + 左 padding
    size_t valid = item_ids.size() > static_cast<size_t>(S) ? static_cast<size_t>(S) : item_ids.size();
    std::vector<int64_t> ids(valid);
    if (valid) ids.assign(item_ids.end() - static_cast<long>(valid), item_ids.end());
    const size_t pad = static_cast<size_t>(S) - valid;

    std::vector<int64_t> padded(static_cast<size_t>(S), 0);
    std::copy(ids.begin(), ids.end(), padded.begin() + static_cast<long>(pad));
    std::vector<uint8_t> mask(static_cast<size_t>(S), 0);
    std::fill(mask.begin() + static_cast<long>(pad), mask.end(), 1);

    // item 查表 → [S, D]，padding 位置（id=0 → 表 0 行）再乘 mask
    std::vector<float> seq_emb = lookup("item", padded);
    for (size_t i = 0; i < static_cast<size_t>(S); ++i)
        if (!mask[i]) std::fill_n(seq_emb.begin() + static_cast<long>(i * D), D, 0.0f);

    // STokenizer：mlp × mask（timestamp_agnostic 合并不加 type_emb、无 sep token）
    Mlp mlp;
    mlp.w1 = s_mlp_w1;
    mlp.w2 = s_mlp_w2;
    Tensor s_tokens({1, S, D});
    mlp.apply(seq_emb.data(), s_tokens.data.data(), S, cfg_.seq_in_dim, D);
    for (int i = 0; i < S; ++i)
        if (!mask[i]) std::fill_n(s_tokens.data.begin() + i * D, D, 0.0f);

    // + pos_emb → RMSNorm（OneTransTokenizer.encode_s）
    Tensor s_emb({1, S, D});
    std::vector<float> tmp(static_cast<size_t>(S * D));
    for (int i = 0; i < S; ++i)
        for (int t = 0; t < D; ++t) tmp[i * D + t] = s_tokens.at(i * D + t) + pos_emb.at(i * D + t);
    rms_norm_rows(tmp.data(), tok_rms_g.data.data(), s_emb.data.data(), S, D, eps);

    Tensor s_mask({1, S});
    for (int i = 0; i < S; ++i) s_mask.at(i) = mask[i] ? 1.0f : 0.0f;
    return {s_emb, s_mask};
}

Tensor EmbeddingFrontend::encode_ns(const ScoreInput& in, const LookupFn& lookup) const {
    const int M = static_cast<int>(in.candidates.size());
    if (M == 0) throw std::runtime_error("score: candidates 为空");
    const int D = cfg_.d_model;
    const int Ns = static_cast<int>(cfg_.ns_group_dims.size());
    const float eps = static_cast<float>(cfg_.rms_eps);

    // dense：用户级 ∥ 候选级 → piecewise
    const int nf = cfg_.piecewise_n_features;
    std::vector<float> dense(static_cast<size_t>(M * nf));
    for (int m = 0; m < M; ++m) {
        const auto& c = in.candidates[m];
        if (in.user_dense.size() != static_cast<size_t>(cfg_.n_user_dense) ||
            c.dense.size() != static_cast<size_t>(cfg_.n_cand_dense))
            throw std::runtime_error("dense 列数不匹配");
        std::copy(in.user_dense.begin(), in.user_dense.end(),
                  dense.begin() + static_cast<long>(m * nf));
        std::copy(c.dense.begin(), c.dense.end(),
                  dense.begin() + static_cast<long>(m * nf + cfg_.n_user_dense));
    }
    Tensor dense_enc = piecewise_forward(pw_w, pw_b, cfg_.piecewise_n_bins, dense.data(), M, nf);

    // uid 查表（同一 user 广播 M 行）
    std::vector<float> uid_row = lookup("user", {in.uid_sparse});
    std::vector<float> uid_emb(static_cast<size_t>(M) * D);
    for (int m = 0; m < M; ++m)
        std::copy(uid_row.begin(), uid_row.end(), uid_emb.begin() + static_cast<long>(m * D));
    // item 查表
    std::vector<int64_t> item_ids;
    item_ids.reserve(M);
    for (const auto& c : in.candidates) item_ids.push_back(c.item_id);
    std::vector<float> item_emb = lookup("item", item_ids);

    // mean-bag（artist/album）：多值查表求均值，空 bag → 0
    auto bag = [&](const std::string& table,
                   const std::vector<std::vector<int64_t>>& ids_per) -> std::vector<float> {
        std::vector<int64_t> flat;
        for (const auto& v : ids_per) flat.insert(flat.end(), v.begin(), v.end());
        std::vector<float> emb = flat.empty() ? std::vector<float>(D, 0.0f) : lookup(table, flat);
        std::vector<float> out(static_cast<size_t>(M * D), 0.0f);
        size_t off = 0;
        for (int m = 0; m < M; ++m) {
            size_t n = ids_per[m].size();
            if (n == 0) continue;
            for (int t = 0; t < D; ++t) {
                float acc = 0.0f;
                for (size_t r = 0; r < n; ++r) acc += emb[(off + r) * D + t];
                out[m * D + t] = acc / static_cast<float>(n);
            }
            off += n;
        }
        return out;
    };
    std::vector<std::vector<int64_t>> artist_ids, album_ids;
    for (const auto& c : in.candidates) {
        artist_ids.push_back(c.artist_ids);
        album_ids.push_back(c.album_ids);
    }
    std::vector<float> artist_emb = bag("artist", artist_ids);
    std::vector<float> album_emb = bag("album", album_ids);

    // NSGroupWise：各组 mlp → stack [M, Ns, D] → RMSNorm
    Tensor ns_emb({M, Ns, D});
    std::vector<const std::vector<float>*> groups = {&dense_enc.data, &uid_emb, &item_emb,
                                                      &artist_emb, &album_emb};
    std::vector<float> stacked(static_cast<size_t>(M * Ns * D));
    std::vector<float> g_tmp(static_cast<size_t>(M) * D);
    for (size_t g = 0; g < cfg_.ns_group_dims.size(); ++g) {
        Mlp mlp;
        mlp.w1 = ns_w1[g];
        mlp.w2 = ns_w2[g];
        // 组 g：[M, in] → [M, D] → 散射到 [M, Ns, D] 的 token g 槽位
        mlp.apply(groups[g]->data(), g_tmp.data(), M, cfg_.ns_group_dims[g], D);
        for (int m = 0; m < M; ++m)
            std::copy_n(g_tmp.begin() + static_cast<long>(m * D), D,
                        stacked.begin() + static_cast<long>((m * Ns + static_cast<int>(g)) * D));
    }
    rms_norm_rows(stacked.data(), tok_rms_g.data.data(), ns_emb.data.data(), M * Ns, D, eps);
    return ns_emb;
}

}  // namespace onetrans
