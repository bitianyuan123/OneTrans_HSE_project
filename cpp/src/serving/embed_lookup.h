// 本地稀疏表（item/user/artist/album embedding），LookupFn 的进程内实现。
//
// 生产部署中该层可替换为 PS HTTP 客户端（参数服务器拉表），接口契约不变：
// lookup(table, ids) → fp32 平铺 [ids.size() * d_model]，行主序。
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/tensor.h"
#include "engine/frontend.h"

namespace onetrans {

class EmbeddingTables {
public:
    static EmbeddingTables load(const ArtifactStore& store);

    // 越界 id 抛异常；返回 [ids.size() * d_model]
    std::vector<float> lookup(const std::string& table, const std::vector<int64_t>& ids) const;

    // 绑定为 EmbeddingFrontend 的 LookupFn
    LookupFn lookup_fn() const {
        return [this](const std::string& t, const std::vector<int64_t>& ids) {
            return this->lookup(t, ids);
        };
    }

    int64_t d_model() const { return d_model_; }

private:
    int64_t d_model_ = 0;
    // table → [rows, d_model] 行主序
    std::unordered_map<std::string, std::vector<float>> tables_;
};

}  // namespace onetrans
