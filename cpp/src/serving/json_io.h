// HTTP/对拍共用的请求体 JSON ↔ 业务输入转换。
//
// 请求体格式即 gen_golden.py 的 cases/*.json（数值对拍锚点）。
#pragma once

#include <string>

#include "common/tensor.h"
#include "engine/frontend.h"

namespace onetrans {

// {"user_id", "item_ids"[], "timestamps"[]} → IngestInput（不等长抛异常）
IngestInput parse_ingest_input(const std::string& body);

// {"user_id", "uid_sparse", "user_dense"[], "candidates"[{item_id,artist_ids[],album_ids[],dense[]}]}
// → ScoreInput（candidates 为空抛异常）
ScoreInput parse_score_input(const std::string& body);

// logits [M, T] → JSON 数组的数组（fp32 全精度）
std::string logits_to_json(const Tensor& logits);

}  // namespace onetrans
