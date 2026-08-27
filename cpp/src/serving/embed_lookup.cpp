#include "serving/embed_lookup.h"

#include <stdexcept>

namespace onetrans {

EmbeddingTables EmbeddingTables::load(const ArtifactStore& store) {
    EmbeddingTables t;
    t.d_model_ = store.config().d_model;

    struct Spec {
        const char* name;
        const char* tensor;
    };
    const Spec specs[] = {
        {"item", "embedder.item_embedding.weight"},
        {"user", "embedder.user_embedding.weight"},
        {"artist", "embedder.artist_embedding.embedding.weight"},
        {"album", "embedder.album_embedding.embedding.weight"},
    };
    for (const auto& s : specs) {
        Tensor w = store.get(s.tensor);
        if (w.shape.size() != 2 || w.shape[1] != t.d_model_)
            throw std::runtime_error(std::string("embedding 表形状异常: ") + s.tensor);
        t.tables_[s.name] = std::move(w.data);
    }
    return t;
}

std::vector<float> EmbeddingTables::lookup(const std::string& table,
                                           const std::vector<int64_t>& ids) const {
    auto it = tables_.find(table);
    if (it == tables_.end()) throw std::runtime_error("未知 embedding 表: " + table);
    const auto& tab = it->second;
    const int64_t rows = static_cast<int64_t>(tab.size()) / d_model_;

    std::vector<float> out;
    out.reserve(ids.size() * static_cast<size_t>(d_model_));
    for (int64_t id : ids) {
        if (id < 0 || id >= rows)
            throw std::runtime_error("embedding id 越界: table=" + table + " id=" + std::to_string(id));
        const float* row = tab.data() + static_cast<size_t>(id) * static_cast<size_t>(d_model_);
        out.insert(out.end(), row, row + d_model_);
    }
    return out;
}

}  // namespace onetrans
