#include "serving/json_io.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "common/json.h"

namespace onetrans {

IngestInput parse_ingest_input(const std::string& body) {
    JsonValue root = json_parse(body);
    IngestInput in;
    in.user_id = root.at("user_id").as_string();
    for (const auto& v : root.at("item_ids").as_array()) in.item_ids.push_back(v.as_int());
    for (const auto& v : root.at("timestamps").as_array()) in.timestamps.push_back(v.as_int());
    if (in.item_ids.size() != in.timestamps.size())
        throw std::runtime_error("item_ids 与 timestamps 必须等长");
    return in;
}

ScoreInput parse_score_input(const std::string& body) {
    JsonValue root = json_parse(body);
    ScoreInput in;
    in.user_id = root.at("user_id").as_string();
    in.uid_sparse = root.at("uid_sparse").as_int();
    for (const auto& v : root.at("user_dense").as_array())
        in.user_dense.push_back(static_cast<float>(v.as_number()));
    for (const auto& c : root.at("candidates").as_array()) {
        CandidateInput cand;
        cand.item_id = c.at("item_id").as_int();
        for (const auto& v : c.at("artist_ids").as_array()) cand.artist_ids.push_back(v.as_int());
        for (const auto& v : c.at("album_ids").as_array()) cand.album_ids.push_back(v.as_int());
        for (const auto& v : c.at("dense").as_array())
            cand.dense.push_back(static_cast<float>(v.as_number()));
        in.candidates.push_back(std::move(cand));
    }
    if (in.candidates.empty()) throw std::runtime_error("candidates 为空");
    return in;
}

std::string logits_to_json(const Tensor& logits) {
    const int64_t M = logits.shape[0], T = logits.shape[1];
    std::ostringstream os;
    os << std::setprecision(9) << "[";
    for (int64_t m = 0; m < M; ++m) {
        if (m) os << ",";
        os << "[";
        for (int64_t t = 0; t < T; ++t) {
            if (t) os << ",";
            os << logits.at(m * T + t);
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

}  // namespace onetrans
