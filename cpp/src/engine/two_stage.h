// 两阶段推理引擎（Nearline prefill / Online 交叉打分）。
//
// 与 onetrans/serving/two_stage.py 数值契约对齐：
// - encode_s（Stage I）：S 侧逐层编码并缓存 (K_s^l, V_s^l)；pyramid 尾部裁剪
//   （token 与 mask 同步）；per_layer_len[l] = 该层输入的有效 token 数（左 padding）。
// - score_ns（Stage II）：读缓存 K/V，广播到 M 候选；S 列按左 padding 重构有效掩码，
//   NS 列按 causal；NS 逐 token 独立投影 + 独立 FFN。
// - score_ns_batch：B 个 (user, 候选) 对打包一次前向（K/V 每层宽度恒定，逐行掩码）。
#pragma once

#include <vector>

#include "engine/model.h"

namespace onetrans {

struct UserKV {
    std::vector<Tensor> k;  // 每层 [1, S_l, H, d]
    std::vector<Tensor> v;
    std::vector<int64_t> per_layer_len;  // 每层输入有效 token 数
    int64_t s_len = 0;                   // 原始有效历史长度
};

class TwoStageRunner {
public:
    explicit TwoStageRunner(const OneTransModel& model) : m_(model) {}

    const OneTransModel& model() const { return m_; }

    // Stage I：s_emb [1, S0, D]，s_mask [1, S0]（0/1 float）
    UserKV encode_s(const Tensor& s_emb, const Tensor& s_mask) const;

    // Stage II：kv 单用户；ns_emb [B=M, Ns, D] → logits [B, T]
    Tensor score_ns(const UserKV& kv, const Tensor& ns_emb) const;

    // Stage II 批量：kvs.size() == ns_emb.shape[0] → logits [B, T]
    Tensor score_ns_batch(const std::vector<const UserKV*>& kvs, const Tensor& ns_emb) const;

private:
    const OneTransModel& m_;
};

}  // namespace onetrans
