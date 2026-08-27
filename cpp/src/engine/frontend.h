// serving 前处理（YambdaEmbedder + OneTransTokenizer 的 C++ 等价实现）。
//
// 与 cpp/tools/gen_golden.py 的 make_s_input / make_ns_input 严格同语义：
// - S 侧：截尾保留最新 max_seq_len 条 → 左 padding（有效靠尾）→ item 查表 →
//   STokenizer（mlp×mask + type_emb + mask + pos_emb + RMSNorm）。
// - NS 侧：dense=[用户级 ∥ 候选级] → piecewise 分箱；uid/item 查表；
//   artist/album mean-bag → NSGroupWiseTokenizer（组顺序 dense/uid/item/artist/album）
//   → RMSNorm。
//
// 稀疏查表经 LookupFn 注入（本地直查 / PS HTTP 客户端均可），保持数据面解耦。
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/tensor.h"
#include "engine/model.h"  // linear_rows / rms_norm_rows 数值原语

namespace onetrans {

struct CandidateInput {
    int64_t item_id = 0;
    std::vector<int64_t> artist_ids;
    std::vector<int64_t> album_ids;
    std::vector<float> dense;  // 候选级 dense（n_cand_dense 列）
};

struct ScoreInput {
    std::string user_id;
    int64_t uid_sparse = 0;
    std::vector<float> user_dense;       // 用户级 dense（n_user_dense 列）
    std::vector<CandidateInput> candidates;
};

struct IngestInput {
    std::string user_id;
    std::vector<int64_t> item_ids;     // 时间升序（最旧在前、最新在尾）
    std::vector<int64_t> timestamps;
};

// 查表函数：表名（item/user/artist/album）+ id 列表 → fp32 平铺 [N * d_model]
using LookupFn = std::function<std::vector<float>(const std::string&, const std::vector<int64_t>&)>;

class EmbeddingFrontend {
public:
    static EmbeddingFrontend load(const ArtifactStore& store);

    // → (s_emb [1, max_seq_len, D], s_mask [1, max_seq_len] 0/1)
    std::pair<Tensor, Tensor> encode_s(const std::vector<int64_t>& item_ids,
                                       const LookupFn& lookup) const;
    // → ns_emb [M, Ns, D]
    Tensor encode_ns(const ScoreInput& in, const LookupFn& lookup) const;

    const ArtifactConfig& config() const { return cfg_; }

private:
    ArtifactConfig cfg_;
    // tokenizer
    Tensor s_mlp_w1, s_mlp_w2;   // [D, seq_in_dim] / [D, D]
    Tensor s_type_emb;            // [1, D]
    Tensor pos_emb;               // [max_seq_len, D]
    Tensor tok_rms_g;             // [D]
    std::vector<Tensor> ns_w1, ns_w2;  // 每组 [D, in] / [D, D]
    // piecewise
    Tensor pw_w, pw_b;            // [n_features, max_n_bins]
};

}  // namespace onetrans
