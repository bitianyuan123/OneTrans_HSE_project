// OneTrans backbone（C++ 实现）与数值原语。
//
// 数值契约（与 onetrans/models/one_trans.py 逐位对齐，golden 对拍验证）：
// - block: z = attn(norm(x)) + x; z = z + ffn(norm(z))（同一个 RMSNorm 权重用两次）
// - 混合参数化：S 段共享投影 W_s，NS 段逐 token 独立投影 W_ns_list[i]
// - 投影输出布局 [rows][3][H][d]：q/k/v 为连续三段 D
// - SDPA：scale = 1/sqrt(head_dim)；全掩行输出 0（对齐 PyTorch SDPA，无 NaN）
// - pyramid：逐层保留尾部（最新）out_seq_num 个 S token
// - FFN：Linear(D→4D) → GELU(exact) → Linear(4D→D)，S 段共享、NS 段逐 token
// - head：末 Ns token 沿 seq 维均值 → Linear(D→T)
#pragma once

#include <functional>
#include <vector>

#include "common/tensor.h"

namespace onetrans {

struct FfnWeights {
    Tensor w1;  // [4D, D]
    Tensor w2;  // [D, 4D]
};

struct AttnWeights {
    Tensor w_s;              // [3D, D]
    std::vector<Tensor> w_ns;  // Ns × [3D, D]
    Tensor final_proj;       // [D, D]
};

struct BlockWeights {
    Tensor norm_g;             // [D]（两次 pre-norm 共享）
    AttnWeights attn;
    FfnWeights ffn_s;          // S 段 FFN
    std::vector<FfnWeights> ffn_ns;  // NS 段逐 token FFN
    int64_t out_seq_num = 0;   // pyramid 该层输出宽度 dims[l+1]
};

class OneTransModel {
public:
    // 从权重目录构建（backbone.* 命名空间）
    static OneTransModel load(const ArtifactStore& store);

    // 单前向（等价性参考/验证用）：tokens [B, L, D]，mask [B, L]（0/1 float）
    Tensor forward(const Tensor& tokens, const Tensor& mask) const;

    ArtifactConfig cfg;
    std::vector<BlockWeights> blocks;
    Tensor head_w;  // [T, D]
    Tensor head_b;  // [T]（nn.Linear 默认 bias=True）
};

// ---------------------------- 数值原语（two_stage 复用） ----------------------------
// RMSNorm 逐行
void rms_norm_rows(const float* x, const float* g, float* out, int64_t rows, int64_t dim, float eps);
// 无 bias 线性：y = x @ w^T；x [n,in]，w [out,in]，y [n,out]
void linear_rows(const float* x, const float* w, float* y, int64_t n, int64_t in, int64_t out);
// FFN：Linear→GELU→Linear（in==out==dim）
void ffn_forward(const FfnWeights& w, const float* x, float* y, int64_t n, int64_t dim);
// 掩码 SDPA：q [Q,H,d]、k/v [K,H,d]、allow [Q*K]（1=允许），out [Q,H,d]。
// 全掩行输出 0。
void sdpa_masked(const float* q, const float* k, const float* v, float* out,
                 int64_t q_rows, int64_t k_rows, int H, int64_t d, const uint8_t* allow);

}  // namespace onetrans
