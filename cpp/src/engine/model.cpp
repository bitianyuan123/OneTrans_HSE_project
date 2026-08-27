#include "engine/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace onetrans {

// --------------------------------------------------------------------------- //
// 数值原语
// --------------------------------------------------------------------------- //

void linear_rows(const float* x, const float* w, float* y, int64_t n, int64_t in, int64_t out) {
    matmul_nt(x, w, y, n, in, out);
}

void rms_norm_rows(const float* x, const float* g, float* out, int64_t rows, int64_t dim, float eps) {
    rms_norm(x, g, out, rows, dim, eps);
}

void ffn_forward(const FfnWeights& w, const float* x, float* y, int64_t n, int64_t dim) {
    int64_t hidden = w.w1.shape[0];
    std::vector<float> h(static_cast<size_t>(n * hidden));
    linear_rows(x, w.w1.data.data(), h.data(), n, dim, hidden);
    for (auto& v : h) v = gelu(v);
    linear_rows(h.data(), w.w2.data.data(), y, n, hidden, dim);
}

void sdpa_masked(const float* q, const float* k, const float* v, float* out,
                 int64_t q_rows, int64_t k_rows, int H, int64_t d, const uint8_t* allow) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int64_t qd = H * d, kd = H * d;
    std::vector<float> scores(static_cast<size_t>(k_rows));

    for (int64_t i = 0; i < q_rows; ++i) {
        const uint8_t* allow_row = allow + i * k_rows;
        for (int h = 0; h < H; ++h) {
            const float* qi = q + i * qd + h * d;
            float m = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int64_t j = 0; j < k_rows; ++j) {
                if (!allow_row[j]) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* kj = k + j * kd + h * d;
                float acc = 0.0f;
                for (int64_t t = 0; t < d; ++t) acc += qi[t] * kj[t];
                float s = acc * scale;
                scores[j] = s;
                if (s > m) m = s;
                any = true;
            }
            float* oi = out + i * qd + h * d;
            if (!any) {
                // 全掩行：输出 0（对齐 PyTorch SDPA，避免 0/0 NaN）
                std::fill(oi, oi + d, 0.0f);
                continue;
            }
            float denom = 0.0f;
            for (int64_t j = 0; j < k_rows; ++j) {
                float e = (scores[j] == -std::numeric_limits<float>::infinity())
                             ? 0.0f
                             : std::exp(scores[j] - m);
                scores[j] = e;
                denom += e;
            }
            float inv = 1.0f / denom;
            std::fill(oi, oi + d, 0.0f);
            for (int64_t j = 0; j < k_rows; ++j) {
                float p = scores[j] * inv;
                if (p == 0.0f) continue;
                const float* vj = v + j * kd + h * d;
                for (int64_t t = 0; t < d; ++t) oi[t] += p * vj[t];
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// 权重加载
// --------------------------------------------------------------------------- //

OneTransModel OneTransModel::load(const ArtifactStore& store) {
    OneTransModel m;
    m.cfg = store.config();
    const int L = m.cfg.num_blocks;
    const int D = m.cfg.d_model;
    const int Ns = m.cfg.ns_tokens_num;

    m.blocks.resize(L);
    for (int l = 0; l < L; ++l) {
        BlockWeights& bw = m.blocks[l];
        bw.out_seq_num = m.cfg.dims[l + 1];
        bw.norm_g = store.get("backbone.blocks." + std::to_string(l) + ".norm.weight");
        bw.attn.w_s = store.get("backbone.blocks." + std::to_string(l) + ".mixed_attn.W_s.weight");
        bw.attn.w_ns.resize(Ns);
        bw.ffn_ns.resize(Ns);
        for (int i = 0; i < Ns; ++i) {
            bw.attn.w_ns[i] = store.get("backbone.blocks." + std::to_string(l) +
                                        ".mixed_attn.W_ns_list." + std::to_string(i) + ".weight");
            bw.ffn_ns[i].w1 = store.get("backbone.blocks." + std::to_string(l) +
                                        ".mixed_ffn.networks_ns_list." + std::to_string(i) + ".0.weight");
            bw.ffn_ns[i].w2 = store.get("backbone.blocks." + std::to_string(l) +
                                        ".mixed_ffn.networks_ns_list." + std::to_string(i) + ".3.weight");
        }
        bw.attn.final_proj =
            store.get("backbone.blocks." + std::to_string(l) + ".mixed_attn.final_proj.weight");
        bw.ffn_s.w1 =
            store.get("backbone.blocks." + std::to_string(l) + ".mixed_ffn.network_s.0.weight");
        bw.ffn_s.w2 =
            store.get("backbone.blocks." + std::to_string(l) + ".mixed_ffn.network_s.3.weight");
    }
    m.head_w = store.get("backbone.linear.weight");
    m.head_b = store.get("backbone.linear.bias");
    return m;
}

// --------------------------------------------------------------------------- //
// 单前向（bshd 语义；B 行独立处理）
// --------------------------------------------------------------------------- //

namespace {

struct Qkv {
    std::vector<float> q, k, v;  // [rows, H, d]
};

// W_s 投影 S 段：x [S, D] → q/k/v [S, H, d]（布局 [S][3][H][d] 切段）
Qkv project_s(const Tensor& w_s, const float* x, int64_t S, int64_t D, int H, int64_t d) {
    Qkv out;
    std::vector<float> qkv(static_cast<size_t>(S * 3 * D));
    linear_rows(x, w_s.data.data(), qkv.data(), S, D, 3 * D);
    out.q.resize(static_cast<size_t>(S * D));
    out.k.resize(static_cast<size_t>(S * D));
    out.v.resize(static_cast<size_t>(S * D));
    for (int64_t s = 0; s < S; ++s) {
        std::copy_n(qkv.begin() + s * 3 * D, D, out.q.begin() + s * D);
        std::copy_n(qkv.begin() + s * 3 * D + D, D, out.k.begin() + s * D);
        std::copy_n(qkv.begin() + s * 3 * D + 2 * D, D, out.v.begin() + s * D);
    }
    return out;
}

}  // namespace

Tensor OneTransModel::forward(const Tensor& tokens, const Tensor& mask) const {
    const int64_t B = tokens.shape[0], L = tokens.shape[1], D = tokens.shape[2];
    const int H = cfg.num_heads, d = D / H;
    const int Ns = cfg.ns_tokens_num;
    const int64_t S = L - Ns;
    const float eps = static_cast<float>(cfg.rms_eps);

    Tensor logits({B, head_w.shape[0]});

    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> x(tokens.data.begin() + b * L * D,
                             tokens.data.begin() + (b + 1) * L * D);
        std::vector<uint8_t> mrow(L);
        for (int64_t j = 0; j < L; ++j) mrow[j] = mask.at(b * L + j) != 0.0f ? 1 : 0;

        int64_t cur_s = S, cur_l = L;
        for (int l = 0; l < cfg.num_blocks; ++l) {
            const BlockWeights& bw = blocks[l];
            // attention
            std::vector<float> h(static_cast<size_t>(cur_l * D));
            rms_norm_rows(x.data(), bw.norm_g.data.data(), h.data(), cur_l, D, eps);

            Qkv sq = project_s(bw.attn.w_s, h.data(), cur_s, D, H, d);
            // NS 逐 token 投影（每 token 一个 [B=1, D] 输入）
            Qkv ns;
            ns.q.resize(static_cast<size_t>(Ns * D));
            ns.k.resize(static_cast<size_t>(Ns * D));
            ns.v.resize(static_cast<size_t>(Ns * D));
            for (int i = 0; i < Ns; ++i) {
                std::vector<float> qkv3(static_cast<size_t>(3 * D));
                linear_rows(h.data() + (cur_s + i) * D, bw.attn.w_ns[i].data.data(), qkv3.data(),
                            1, D, 3 * D);
                std::copy_n(qkv3.begin(), D, ns.q.begin() + i * D);
                std::copy_n(qkv3.begin() + D, D, ns.k.begin() + i * D);
                std::copy_n(qkv3.begin() + 2 * D, D, ns.v.begin() + i * D);
            }
            // 拼接 K/V：[L, H, d]
            std::vector<float> kk(static_cast<size_t>(cur_l * D)), vv(static_cast<size_t>(cur_l * D));
            std::copy(sq.k.begin(), sq.k.end(), kk.begin());
            std::copy(ns.k.begin(), ns.k.end(), kk.begin() + cur_s * D);
            std::copy(sq.v.begin(), sq.v.end(), vv.begin());
            std::copy(ns.v.begin(), ns.v.end(), vv.begin() + cur_s * D);

            // 掩码：padding 列 + causal
            std::vector<uint8_t> allow(static_cast<size_t>(cur_l * cur_l), 1);
            for (int64_t i = 0; i < cur_l; ++i)
                for (int64_t j = 0; j < cur_l; ++j)
                    if (j > i || !mrow[j]) allow[static_cast<size_t>(i * cur_l + j)] = 0;

            // 拼接 Q：[L, H, d]
            std::vector<float> qq(static_cast<size_t>(cur_l * D));
            std::copy(sq.q.begin(), sq.q.end(), qq.begin());
            std::copy(ns.q.begin(), ns.q.end(), qq.begin() + cur_s * D);

            std::vector<float> attn(static_cast<size_t>(cur_l * D));
            sdpa_masked(qq.data(), kk.data(), vv.data(), attn.data(), cur_l, cur_l, H, d,
                        allow.data());
            std::vector<float> proj(static_cast<size_t>(cur_l * D));
            linear_rows(attn.data(), bw.attn.final_proj.data.data(), proj.data(), cur_l, D, D);
            std::vector<float> z(static_cast<size_t>(cur_l * D));
            for (size_t i = 0; i < z.size(); ++i) z[i] = proj[i] + x[i];

            // FFN（S 段共享 / NS 段逐 token）——norm 共享同一权重
            std::vector<float> h2(static_cast<size_t>(cur_l * D));
            rms_norm_rows(z.data(), bw.norm_g.data.data(), h2.data(), cur_l, D, eps);
            std::vector<float> f(static_cast<size_t>(cur_l * D));
            ffn_forward(bw.ffn_s, h2.data(), f.data(), cur_s, D);
            for (int i = 0; i < Ns; ++i) {
                ffn_forward(bw.ffn_ns[i], h2.data() + (cur_s + i) * D, f.data() + (cur_s + i) * D,
                            1, D);
            }
            for (size_t i = 0; i < z.size(); ++i) z[i] += f[i];

            // pyramid：保留尾部 out_seq_num 个 S token + 全部 NS
            int64_t keep = bw.out_seq_num;
            int64_t new_s = keep;
            std::vector<float> nx(static_cast<size_t>((new_s + Ns) * D));
            std::vector<uint8_t> nm(static_cast<size_t>(new_s + Ns));
            std::copy_n(z.begin() + (cur_s - keep) * D, (new_s + Ns) * D, nx.begin());
            std::copy_n(mrow.begin() + (cur_s - keep), new_s + Ns, nm.begin());
            x = std::move(nx);
            mrow = std::move(nm);
            cur_s = new_s;
            cur_l = new_s + Ns;
        }

        // head：末 Ns token 均值 → [D] → linear
        std::vector<float> pooled(static_cast<size_t>(D), 0.0f);
        for (int i = 0; i < Ns; ++i)
            for (int64_t t = 0; t < D; ++t) pooled[t] += x[(cur_s + i) * D + t];
        for (int64_t t = 0; t < D; ++t) pooled[t] /= static_cast<float>(Ns);
        std::vector<float> lg(static_cast<size_t>(head_w.shape[0]));
        linear_rows(pooled.data(), head_w.data.data(), lg.data(), 1, D, head_w.shape[0]);
        for (int64_t t = 0; t < head_w.shape[0]; ++t)
            logits.at(b * head_w.shape[0] + t) = lg[t] + head_b.at(t);
    }
    return logits;
}

}  // namespace onetrans
