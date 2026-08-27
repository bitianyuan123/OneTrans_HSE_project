// 通用张量（fp32、行主序）与张量目录（manifest.json + *.bin）加载。
//
// 与 Python 侧导出格式（cpp/tools/export_weights.py）对齐：
//   manifest.tensors[name] = {shape, offset, dtype}，offset 单位为 float 元素，
//   所有张量在 .bin 中按声明顺序 fp32 小端连续平铺。
#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace onetrans {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shp) : shape(std::move(shp)) {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        data.assign(static_cast<size_t>(n), 0.0f);
    }
    static Tensor zeros(std::vector<int64_t> shp) { return Tensor(std::move(shp)); }

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    float& at(int64_t i) { return data[static_cast<size_t>(i)]; }
    const float& at(int64_t i) const { return data[static_cast<size_t>(i)]; }
};

// 线性代数原语（无 bias；PyTorch nn.Linear 语义 y = x @ W^T）
// x: [n, in]，W: [out, in]，y: [n, out]
void matmul_nt(const float* x, const float* w, float* y, int64_t n, int64_t in, int64_t out);

// RMSNorm：out = x * rsqrt(mean(x^2) + eps) * g（逐行）
void rms_norm(const float* x, const float* g, float* out, int64_t rows, int64_t dim, float eps);

// 精确 GELU：0.5x(1 + erf(x / sqrt(2)))，对齐 torch.nn.GELU 默认
inline float gelu(float v) {
    return 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
}

struct TensorMeta {
    std::string name;
    std::vector<int64_t> shape;
    int64_t offset;  // 元素偏移
};

struct ArtifactConfig {
    int d_model = 128;
    int num_heads = 4;
    int num_blocks = 4;
    int max_seq_len = 50;
    int min_seq_len = 5;
    int ns_tokens_num = 5;
    bool use_cls_token = false;
    std::vector<int64_t> dims;  // pyramid 逐层宽度（num_blocks + 1 个）
    double rms_eps = 1.1920928955078125e-07;
    int seq_in_dim = 128;
    std::vector<int64_t> ns_group_dims;
    int num_items = 0, num_users = 0, num_artists = 0, num_albums = 0;
    int n_user_dense = 15, n_cand_dense = 15;
    int piecewise_n_features = 30;
    int piecewise_max_n_bins = 8;
    std::vector<int64_t> piecewise_n_bins;  // 每列 bin 数
};

// manifest.json + weights.bin（或 golden.bin）的只读目录。
class ArtifactStore {
public:
    // dir 下读 manifest.json 与 bin_file；不存在则抛异常。
    void load(const std::string& dir, const std::string& bin_file);

    bool has(const std::string& name) const;
    Tensor get(const std::string& name) const;          // 拷贝出张量
    const std::vector<float>& raw() const { return blob_; }
    const TensorMeta& meta(const std::string& name) const;
    const ArtifactConfig& config() const { return cfg_; }

private:
    ArtifactConfig cfg_;
    std::vector<float> blob_;
    std::vector<TensorMeta> metas_;
};

}  // namespace onetrans
