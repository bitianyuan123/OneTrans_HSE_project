// 数值对拍：C++ 全链路 vs Python 参考实现（gen_golden.py 产出的 golden）。
//
// 覆盖层次（逐层锚定，失败可定位）：
//   1. frontend.encode_s   vs golden s_emb / s_mask（前处理 S 侧）
//   2. runner.encode_s     vs golden k_l / v_l（Stage I 逐层 KV）
//   3. frontend.encode_ns  vs golden ns_emb（前处理 NS 侧）
//   4. runner.score_ns     vs golden logits_two_stage（Stage II 两阶段）
//   5. model.forward       vs golden logits_single（单前向等价性）
//   6. KV serialize roundtrip + LocalKVStore put/get + Router 一致性（工程正确性）
//
// 运行：./verify_golden --weights cpp/artifacts/weights --golden cpp/artifacts/golden
// 退出码：0 = 全部通过；1 = 存在超差项。
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/json.h"
#include "common/tensor.h"
#include "engine/frontend.h"
#include "engine/model.h"
#include "engine/two_stage.h"
#include "kv/router.h"
#include "kv/serialize.h"
#include "kv/store.h"
#include "serving/embed_lookup.h"
#include "serving/json_io.h"

using namespace onetrans;

namespace {

struct CheckResult {
    std::string name;
    double max_diff = 0.0;
    double tol = 0.0;
    bool pass = false;
    int64_t n = 0;
};

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, int64_t* n) {
    if (a.size() != b.size()) throw std::runtime_error("对拍张量长度不一致");
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double v = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (v > d) d = v;
    }
    *n = static_cast<int64_t>(a.size());
    return d;
}

CheckResult check(const std::string& name, const Tensor& got, const Tensor& want, double tol) {
    CheckResult r;
    r.name = name;
    r.tol = tol;
    if (got.shape != want.shape) {
        std::cerr << "[FAIL] " << name << ": 形状不一致\n";
        return r;
    }
    r.max_diff = max_abs_diff(got.data, want.data, &r.n);
    r.pass = r.max_diff <= tol;
    return r;
}

CheckResult check_exact(const std::string& name, bool ok) {
    CheckResult r;
    r.name = name;
    r.max_diff = ok ? 0.0 : 1.0;
    r.tol = 0.0;
    r.pass = ok;
    r.n = 1;
    return r;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("文件打开失败: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void report(const std::vector<CheckResult>& results) {
    int failed = 0;
    std::printf("%-28s %14s %10s %8s\n", "CHECK", "MAX|DIFF|", "TOL", "RESULT");
    std::printf("%s\n", std::string(64, '-').c_str());
    for (const auto& r : results) {
        std::printf("%-28s %14.3e %10.1e %8s\n", r.name.c_str(), r.max_diff, r.tol,
                    r.pass ? "PASS" : "FAIL");
        if (!r.pass) ++failed;
    }
    std::printf("%s\n", std::string(64, '-').c_str());
    std::printf("共 %zu 项，失败 %d 项\n", results.size(), failed);
}

}  // namespace

int main(int argc, char** argv) {
    std::string weights_dir = "cpp/artifacts/weights";
    std::string golden_dir = "cpp/artifacts/golden";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--weights" && i + 1 < argc) weights_dir = argv[++i];
        else if (a == "--golden" && i + 1 < argc) golden_dir = argv[++i];
        else {
            std::cerr << "用法: verify_golden [--weights DIR] [--golden DIR]\n";
            return 2;
        }
    }

    try {
        // ---- 装配 ---- //
        ArtifactStore wstore;
        wstore.load(weights_dir, "weights.bin");
        OneTransModel model = OneTransModel::load(wstore);
        EmbeddingFrontend frontend = EmbeddingFrontend::load(wstore);
        EmbeddingTables tables = EmbeddingTables::load(wstore);
        TwoStageRunner runner(model);
        LookupFn lookup = tables.lookup_fn();

        ArtifactStore golden;
        golden.load(golden_dir, "golden.bin");
        JsonValue case_meta = json_parse(read_file(golden_dir + "/manifest.json")).at("case_meta");

        IngestInput ingest = parse_ingest_input(read_file(golden_dir + "/cases/ingest_case.json"));
        ScoreInput score = parse_score_input(read_file(golden_dir + "/cases/score_case.json"));

        std::vector<CheckResult> results;

        // ---- 1. 前处理 S 侧 ---- //
        auto [s_emb, s_mask] = frontend.encode_s(ingest.item_ids, lookup);
        results.push_back(check("frontend.encode_s/s_emb", s_emb, golden.get("s_emb"), 1e-4));
        results.push_back(check("frontend.encode_s/s_mask", s_mask, golden.get("s_mask"), 0.0));

        // ---- 2. Stage I 逐层 KV ---- //
        UserKV kv = runner.encode_s(s_emb, s_mask);
        bool meta_ok = kv.s_len == case_meta.at("s_len").as_int();
        for (size_t l = 0; l < kv.per_layer_len.size(); ++l)
            meta_ok = meta_ok && kv.per_layer_len[l] ==
                                     case_meta.at("per_layer_len").as_array()[l].as_int();
        results.push_back(check_exact("encode_s/per_layer_len", meta_ok));
        for (size_t l = 0; l < kv.k.size(); ++l) {
            results.push_back(check("encode_s/k_" + std::to_string(l), kv.k[l],
                                    golden.get("k_" + std::to_string(l)), 1e-4));
            results.push_back(check("encode_s/v_" + std::to_string(l), kv.v[l],
                                    golden.get("v_" + std::to_string(l)), 1e-4));
        }

        // ---- 3. 前处理 NS 侧 ---- //
        Tensor ns_emb = frontend.encode_ns(score, lookup);
        results.push_back(check("frontend.encode_ns/ns_emb", ns_emb, golden.get("ns_emb"), 1e-4));

        // ---- 4. Stage II 两阶段 ---- //
        Tensor logits2 = runner.score_ns(kv, ns_emb);
        results.push_back(
            check("score_ns/logits_two_stage", logits2, golden.get("logits_two_stage"), 5e-4));

        // ---- 4b. 逐 block NS 中间量（对拍定位） ---- //
        {
            // 复刻 score_ns_impl 的逐 block 状态（与 two_stage.cpp 同语义）
            const int64_t B = ns_emb.shape[0];
            const int64_t Ns = ns_emb.shape[1];
            const int64_t D = ns_emb.shape[2];
            const int H = model.cfg.num_heads;
            const int64_t hd = D / H;
            const float eps = static_cast<float>(model.cfg.rms_eps);
            std::vector<float> ns(ns_emb.data);
            std::vector<const UserKV*> kvs(static_cast<size_t>(B), &kv);
            for (int l = 0; l < model.cfg.num_blocks; ++l) {
                const BlockWeights& bw = model.blocks[l];
                const int64_t S_l = kvs[0]->k[l].shape[1];
                std::vector<float> h(static_cast<size_t>(B * Ns * D));
                rms_norm_rows(ns.data(), bw.norm_g.data.data(), h.data(), B * Ns, D, eps);
                // NS 逐 token 投影（与 two_stage.cpp project_ns 同语义）
                struct Qkv3 {
                    std::vector<float> q, k, v;
                } nsq{std::vector<float>(static_cast<size_t>(B * Ns * D)),
                     std::vector<float>(static_cast<size_t>(B * Ns * D)),
                     std::vector<float>(static_cast<size_t>(B * Ns * D))};
                {
                    std::vector<float> qkv3(static_cast<size_t>(3 * D));
                    for (int64_t b = 0; b < B; ++b)
                        for (int64_t i = 0; i < Ns; ++i) {
                            const float* x = h.data() + (b * Ns + i) * D;
                            linear_rows(x, bw.attn.w_ns[i].data.data(), qkv3.data(), 1, D, 3 * D);
                            std::copy_n(qkv3.begin(), D, nsq.q.begin() + (b * Ns + i) * D);
                            std::copy_n(qkv3.begin() + D, D, nsq.k.begin() + (b * Ns + i) * D);
                            std::copy_n(qkv3.begin() + 2 * D, D, nsq.v.begin() + (b * Ns + i) * D);
                        }
                }
                std::vector<float> kk(static_cast<size_t>(B * (S_l + Ns) * D));
                std::vector<float> vv(static_cast<size_t>(B * (S_l + Ns) * D));
                for (int64_t b = 0; b < B; ++b) {
                    std::copy(kv.k[l].data.begin(), kv.k[l].data.end(),
                              kk.begin() + b * (S_l + Ns) * D);
                    std::copy(nsq.k.begin() + b * Ns * D, nsq.k.begin() + (b + 1) * Ns * D,
                              kk.begin() + b * (S_l + Ns) * D + S_l * D);
                    std::copy(kv.v[l].data.begin(), kv.v[l].data.end(),
                              vv.begin() + b * (S_l + Ns) * D);
                    std::copy(nsq.v.begin() + b * Ns * D, nsq.v.begin() + (b + 1) * Ns * D,
                              vv.begin() + b * (S_l + Ns) * D + S_l * D);
                }
                std::vector<uint8_t> allow(static_cast<size_t>(B * Ns * (S_l + Ns)), 1);
                for (int64_t b = 0; b < B; ++b) {
                    int64_t valid = kv.per_layer_len[l];
                    for (int64_t i = 0; i < Ns; ++i) {
                        uint8_t* row = allow.data() + (b * Ns + i) * (S_l + Ns);
                        for (int64_t j = 0; j < S_l; ++j)
                            if (j < S_l - valid) row[j] = 0;
                        for (int64_t j = S_l; j < S_l + Ns; ++j)
                            if (j - S_l > i) row[j] = 0;
                    }
                }
                std::vector<float> attn(static_cast<size_t>(B * Ns * D));
                for (int64_t b = 0; b < B; ++b) {
                    sdpa_masked(nsq.q.data() + b * Ns * D, kk.data() + b * (S_l + Ns) * D,
                                vv.data() + b * (S_l + Ns) * D, attn.data() + b * Ns * D, Ns,
                                S_l + Ns, H, hd, allow.data() + b * Ns * (S_l + Ns));
                }
                std::vector<float> proj(static_cast<size_t>(B * Ns * D));
                linear_rows(attn.data(), bw.attn.final_proj.data.data(), proj.data(), B * Ns, D, D);
                std::vector<float> z(static_cast<size_t>(B * Ns * D));
                for (size_t i = 0; i < z.size(); ++i) z[i] = proj[i] + ns[i];
                std::vector<float> h2(static_cast<size_t>(B * Ns * D));
                rms_norm_rows(z.data(), bw.norm_g.data.data(), h2.data(), B * Ns, D, eps);
                std::vector<float> f(static_cast<size_t>(B * Ns * D));
                for (int64_t b = 0; b < B; ++b)
                    for (int64_t i = 0; i < Ns; ++i)
                        ffn_forward(bw.ffn_ns[i], h2.data() + (b * Ns + i) * D,
                                    f.data() + (b * Ns + i) * D, 1, D);
                for (size_t i = 0; i < z.size(); ++i) z[i] += f[i];
                ns = z;
                Tensor dbg({B, Ns, D});
                std::copy(ns.begin(), ns.end(), dbg.data.begin());
                if (golden.has("dbg_ns_" + std::to_string(l)))
                    results.push_back(check("dbg/ns_after_block_" + std::to_string(l), dbg,
                                            golden.get("dbg_ns_" + std::to_string(l)), 1e-4));
            }
        }

        // ---- 5. 单前向等价性（tokens = [s_emb ∥ ns_emb] 广播到 M） ---- //
        const int64_t M = ns_emb.shape[0], S0 = s_emb.shape[1], Ns = ns_emb.shape[1],
                      D = s_emb.shape[2];
        Tensor tokens({M, S0 + Ns, D}), mask({M, S0 + Ns});
        for (int64_t b = 0; b < M; ++b) {
            std::copy(s_emb.data.begin(), s_emb.data.end(),
                      tokens.data.begin() + b * (S0 + Ns) * D);
            std::copy(ns_emb.data.begin() + b * Ns * D, ns_emb.data.begin() + (b + 1) * Ns * D,
                      tokens.data.begin() + b * (S0 + Ns) * D + S0 * D);
            for (int64_t j = 0; j < S0; ++j) mask.at(b * (S0 + Ns) + j) = s_mask.at(j);
            for (int64_t j = 0; j < Ns; ++j) mask.at(b * (S0 + Ns) + S0 + j) = 1.0f;
        }
        Tensor logits1 = model.forward(tokens, mask);
        results.push_back(
            check("forward/logits_single", logits1, golden.get("logits_single"), 5e-4));

        // ---- 6. 工程正确性：序列化 roundtrip / KVStore / Router ---- //
        std::string payload = kv_serialize(kv);
        KvPayload hdr = kv_read_header(payload);
        bool hdr_ok = hdr.s_len == kv.s_len && hdr.n_layers == static_cast<int>(kv.k.size()) &&
                      hdr.dtype == "float32";
        for (size_t l = 0; l < kv.per_layer_len.size(); ++l)
            hdr_ok = hdr_ok && hdr.per_layer_len[l] == kv.per_layer_len[l];
        results.push_back(check_exact("kv_serialize/header", hdr_ok));

        UserKV kv2 = kv_deserialize(payload);
        bool rt_ok = kv2.s_len == kv.s_len && kv2.k.size() == kv.k.size();
        for (size_t l = 0; l < kv.k.size(); ++l)
            rt_ok = rt_ok && kv2.k[l].shape == kv.k[l].shape && kv2.v[l].shape == kv.v[l].shape &&
                    kv2.k[l].data == kv.k[l].data && kv2.v[l].data == kv.v[l].data;
        results.push_back(check_exact("kv_roundtrip/tensors", rt_ok));

        // concat：base + delta 逐层拼接后层数不变、s_len 相加
        std::string cat = kv_concat_payload(payload, payload);
        KvPayload cat_hdr = kv_read_header(cat);
        bool cat_ok = cat_hdr.n_layers == hdr.n_layers && cat_hdr.s_len == 2 * hdr.s_len &&
                      cat_hdr.per_layer_len[0] == 2 * hdr.per_layer_len[0];
        results.push_back(check_exact("kv_concat/payload", cat_ok));

        // LocalKVStore put/get/mget/del
        LocalKVStore store;
        UserKVRecord rec;
        rec.key = KVKey{"v42", "u-001"};
        rec.s_len = kv.s_len;
        rec.per_layer_len = kv.per_layer_len;
        rec.payload = payload;
        rec.created_at = 12345;
        store.put(rec);
        auto got = store.get(KVKey{"v42", "u-001"});
        bool store_ok = got && got->payload == payload && got->checksum() == rec.checksum() &&
                        store.size() == 1 && !store.get(KVKey{"v42", "ghost"}) &&
                        store.del({KVKey{"v42", "u-001"}}) == 1 && store.size() == 0;
        results.push_back(check_exact("kvstore/put_get_del", store_ok));

        // Router：jump hash 稳定性 + 扩容 remap 比例（一致性哈希核心性质）
        JumpConsistentHash r3(3), r4(4);
        std::vector<int> a3, a4;
        int changed = 0;
        const int N = 200;
        for (int i = 0; i < N; ++i) {
            std::string u = "user-" + std::to_string(i);
            int s3 = r3.shard_of(u), s4 = r4.shard_of(u);
            a3.push_back(s3);
            a4.push_back(s4);
            changed += (s3 != s4);
            if (s3 < 0 || s3 >= 3 || s4 < 0 || s4 >= 4) {
                changed = N + 1;  // 越界 → 强制失败
                break;
            }
            if (r3.shard_of(u) != s3) {  // 确定性
                changed = N + 1;
                break;
            }
        }
        // 3→4 分片：理论 remap ≈ 1/4（跳变哈希最优）；阈值放宽到 0.5 排除实现错误
        double remap = static_cast<double>(changed) / N;
        results.push_back(check_exact("router/jump_remap_ratio", remap >= 0.0 && remap < 0.5));

        RingHash ring;
        ring.add_node("w0");
        ring.add_node("w1");
        ring.add_node("w2");
        bool ring_ok = ring.shard_of("user-1").size() >= 2 && ring.node_count() == 3;
        ring.remove_node("w2");
        ring_ok = ring_ok && ring.node_count() == 2 && ring.shard_of("user-1") != "w2";
        results.push_back(check_exact("router/ring_add_remove", ring_ok));

        report(results);
        for (const auto& r : results)
            if (!r.pass) return 1;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[verify_golden] 异常: " << e.what() << "\n";
        return 1;
    }
}
