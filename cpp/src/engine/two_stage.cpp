#include "engine/two_stage.h"

#include <stdexcept>

namespace onetrans {

namespace {

struct LayerQkv {
    // NS 段 Q/K/V：[B, Ns, H*d]（bshd 去 batch 维）
    std::vector<float> q, k, v;
};

// NS 逐 token 投影：h [B, Ns, D] → q/k/v [B, Ns, H*d]
// 语义：token i 的所有候选行经 W_ns[i] 投影（Python _project_ns）。
LayerQkv project_ns(const AttnWeights& attn, const float* h, int64_t B, int64_t Ns, int64_t D,
                    int H, int64_t d) {
    LayerQkv out;
    out.q.resize(static_cast<size_t>(B * Ns * D));
    out.k.resize(static_cast<size_t>(B * Ns * D));
    out.v.resize(static_cast<size_t>(B * Ns * D));
    std::vector<float> qkv3(static_cast<size_t>(3 * D));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < Ns; ++i) {
            // token i 的候选 b 输入行
            const float* x = h + (b * Ns + i) * D;
            linear_rows(x, attn.w_ns[i].data.data(), qkv3.data(), 1, D, 3 * D);
            float* q = out.q.data() + (b * Ns + i) * D;
            float* k = out.k.data() + (b * Ns + i) * D;
            float* v = out.v.data() + (b * Ns + i) * D;
            std::copy_n(qkv3.begin(), D, q);
            std::copy_n(qkv3.begin() + D, D, k);
            std::copy_n(qkv3.begin() + 2 * D, D, v);
        }
    }
    return out;
}

// NS 段逐 token FFN：h [B, Ns, D] → out [B, Ns, D]
void apply_ns_ffn(const std::vector<FfnWeights>& ffn_ns, const float* h, float* out, int64_t B,
                  int64_t Ns, int64_t D) {
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < Ns; ++i)
            ffn_forward(ffn_ns[i], h + (b * Ns + i) * D, out + (b * Ns + i) * D, 1, D);
}

}  // namespace

// --------------------------------------------------------------------------- //
// Stage I
// --------------------------------------------------------------------------- //

UserKV TwoStageRunner::encode_s(const Tensor& s_emb, const Tensor& s_mask) const {
    if (s_emb.shape[0] != 1) throw std::runtime_error("encode_s 只支持 B=1");
    const int64_t D = m_.cfg.d_model;
    const int H = m_.cfg.num_heads, d = D / H;
    const float eps = static_cast<float>(m_.cfg.rms_eps);

    int64_t S = s_emb.shape[1];
    std::vector<float> s(s_emb.data);
    std::vector<uint8_t> smask(S);
    int64_t s_len = 0;
    for (int64_t j = 0; j < S; ++j) {
        smask[j] = s_mask.at(j) != 0.0f ? 1 : 0;
        s_len += smask[j];
    }

    UserKV kv;
    kv.s_len = s_len;

    for (int l = 0; l < m_.cfg.num_blocks; ++l) {
        const BlockWeights& bw = m_.blocks[l];
        int64_t valid = 0;
        for (auto v : smask) valid += v;
        kv.per_layer_len.push_back(valid);

        std::vector<float> h(static_cast<size_t>(S * D));
        rms_norm_rows(s.data(), bw.norm_g.data.data(), h.data(), S, D, eps);

        // W_s 投影 → q/k/v [S, H, d]
        std::vector<float> qkv(static_cast<size_t>(S * 3 * D));
        linear_rows(h.data(), bw.attn.w_s.data.data(), qkv.data(), S, D, 3 * D);
        Tensor k_t({1, S, H, d}), v_t({1, S, H, d});
        std::vector<float> q(static_cast<size_t>(S * D));
        for (int64_t r = 0; r < S; ++r) {
            std::copy_n(qkv.begin() + r * 3 * D, D, q.begin() + r * D);
            std::copy_n(qkv.begin() + r * 3 * D + D, D, k_t.data.begin() + r * D);
            std::copy_n(qkv.begin() + r * 3 * D + 2 * D, D, v_t.data.begin() + r * D);
        }
        kv.k.push_back(std::move(k_t));
        kv.v.push_back(std::move(v_t));

        // S 自注意力：padding 列掩 + causal
        std::vector<uint8_t> allow(static_cast<size_t>(S * S), 1);
        for (int64_t i = 0; i < S; ++i)
            for (int64_t j = 0; j < S; ++j)
                if (j > i || !smask[j]) allow[static_cast<size_t>(i * S + j)] = 0;

        std::vector<float> attn(static_cast<size_t>(S * D));
        sdpa_masked(q.data(), kv.k[l].data.data(), kv.v[l].data.data(), attn.data(), S, S, H, d,
                    allow.data());
        std::vector<float> proj(static_cast<size_t>(S * D));
        linear_rows(attn.data(), bw.attn.final_proj.data.data(), proj.data(), S, D, D);

        std::vector<float> z(static_cast<size_t>(S * D));
        for (size_t i = 0; i < z.size(); ++i) z[i] = proj[i] + s[i];

        std::vector<float> h2(static_cast<size_t>(S * D));
        rms_norm_rows(z.data(), bw.norm_g.data.data(), h2.data(), S, D, eps);
        std::vector<float> f(static_cast<size_t>(S * D));
        ffn_forward(bw.ffn_s, h2.data(), f.data(), S, D);
        for (size_t i = 0; i < z.size(); ++i) z[i] += f[i];

        // pyramid 尾部裁剪（token 与 mask 同步）
        int64_t keep = bw.out_seq_num;
        s.assign(z.begin() + static_cast<long>((S - keep) * D), z.end());
        smask.assign(smask.begin() + (S - keep), smask.end());
        S = keep;
    }
    return kv;
}

// --------------------------------------------------------------------------- //
// Stage II（单用户多候选 + 批量共用实现）
// --------------------------------------------------------------------------- //

namespace {

Tensor score_ns_impl(const OneTransModel& m, const std::vector<const UserKV*>& kvs,
                     const Tensor& ns_emb) {
    const int64_t B = ns_emb.shape[0];
    if (static_cast<int64_t>(kvs.size()) != B)
        throw std::runtime_error("kvs 数量须等于 ns_emb batch");
    const int64_t Ns = ns_emb.shape[1];
    const int64_t D = ns_emb.shape[2];
    const int H = m.cfg.num_heads, d = D / H;
    const float eps = static_cast<float>(m.cfg.rms_eps);

    std::vector<float> ns(ns_emb.data);
    const int64_t T = m.head_w.shape[0];

    for (int l = 0; l < m.cfg.num_blocks; ++l) {
        const BlockWeights& bw = m.blocks[l];
        const int64_t S_l = kvs[0]->k[l].shape[1];  // pyramid 宽度恒定

        std::vector<float> h(static_cast<size_t>(B * Ns * D));
        rms_norm_rows(ns.data(), bw.norm_g.data.data(), h.data(), B * Ns, D, eps);
        LayerQkv nsq = project_ns(bw.attn, h.data(), B, Ns, D, H, d);

        // K/V 拼接：每候选 b：[k_s^l ∥ k_ns(b)] → [B, S_l+Ns, H, d]
        std::vector<float> kk(static_cast<size_t>(B * (S_l + Ns) * D));
        std::vector<float> vv(static_cast<size_t>(B * (S_l + Ns) * D));
        for (int64_t b = 0; b < B; ++b) {
            const UserKV* kv = kvs[b];
            std::copy(kv->k[l].data.begin(), kv->k[l].data.end(),
                      kk.begin() + b * (S_l + Ns) * D);
            std::copy(nsq.k.begin() + b * Ns * D, nsq.k.begin() + (b + 1) * Ns * D,
                      kk.begin() + b * (S_l + Ns) * D + S_l * D);
            std::copy(kv->v[l].data.begin(), kv->v[l].data.end(),
                      vv.begin() + b * (S_l + Ns) * D);
            std::copy(nsq.v.begin() + b * Ns * D, nsq.v.begin() + (b + 1) * Ns * D,
                      vv.begin() + b * (S_l + Ns) * D + S_l * D);
        }

        // 逐候选左 padding 有效掩码 + NS causal
        std::vector<uint8_t> allow(static_cast<size_t>(B * Ns * (S_l + Ns)), 1);
        for (int64_t b = 0; b < B; ++b) {
            int64_t valid = kvs[b]->per_layer_len[l];
            for (int64_t i = 0; i < Ns; ++i) {
                uint8_t* row = allow.data() + (b * Ns + i) * (S_l + Ns);
                for (int64_t j = 0; j < S_l; ++j)
                    if (j < S_l - valid) row[j] = 0;  // 左 padding 列
                for (int64_t j = S_l; j < S_l + Ns; ++j)
                    if (j - S_l > i) row[j] = 0;  // NS causal
            }
        }

        // 分候选 SDPA（Q 布局 [b, Ns, H, d] → 每候选一个 [Ns, H, d] 块）
        std::vector<float> attn(static_cast<size_t>(B * Ns * D));
        for (int64_t b = 0; b < B; ++b) {
            sdpa_masked(nsq.q.data() + b * Ns * D, kk.data() + b * (S_l + Ns) * D,
                        vv.data() + b * (S_l + Ns) * D, attn.data() + b * Ns * D, Ns, S_l + Ns,
                        H, d, allow.data() + b * Ns * (S_l + Ns));
        }
        std::vector<float> proj(static_cast<size_t>(B * Ns * D));
        linear_rows(attn.data(), bw.attn.final_proj.data.data(), proj.data(), B * Ns, D, D);

        std::vector<float> z(static_cast<size_t>(B * Ns * D));
        for (size_t i = 0; i < z.size(); ++i) z[i] = proj[i] + ns[i];

        std::vector<float> h2(static_cast<size_t>(B * Ns * D));
        rms_norm_rows(z.data(), bw.norm_g.data.data(), h2.data(), B * Ns, D, eps);
        std::vector<float> f(static_cast<size_t>(B * Ns * D));
        apply_ns_ffn(bw.ffn_ns, h2.data(), f.data(), B, Ns, D);
        for (size_t i = 0; i < z.size(); ++i) z[i] += f[i];
        ns = std::move(z);
    }

    // head：末 Ns token 均值 → linear（含 bias）
    Tensor logits({B, T});
    std::vector<float> pooled(static_cast<size_t>(D));
    for (int64_t b = 0; b < B; ++b) {
        std::fill(pooled.begin(), pooled.end(), 0.0f);
        for (int64_t i = 0; i < Ns; ++i)
            for (int64_t t = 0; t < D; ++t) pooled[t] += ns[(b * Ns + i) * D + t];
        for (int64_t t = 0; t < D; ++t) pooled[t] /= static_cast<float>(Ns);
        linear_rows(pooled.data(), m.head_w.data.data(), logits.data.data() + b * T, 1, D, T);
        for (int64_t t = 0; t < T; ++t) logits.data.data()[b * T + t] += m.head_b.at(t);
    }
    return logits;
}

}  // namespace

Tensor TwoStageRunner::score_ns(const UserKV& kv, const Tensor& ns_emb) const {
    const int64_t B = ns_emb.shape[0];
    std::vector<const UserKV*> kvs(static_cast<size_t>(B), &kv);
    return score_ns_impl(m_, kvs, ns_emb);
}

Tensor TwoStageRunner::score_ns_batch(const std::vector<const UserKV*>& kvs,
                                       const Tensor& ns_emb) const {
    return score_ns_impl(m_, kvs, ns_emb);
}

}  // namespace onetrans
