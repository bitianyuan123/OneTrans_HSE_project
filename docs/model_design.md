# 序列 Transformer 精排 · 模型设计与正确性验证

> 版本：v1.0
> 文档类别：**① 模型层**——模型结构、计算逻辑、训练、两阶段拆分与等价性、正确性验证方法学。
> 边界约定：本文只谈「算法/模型是什么、为什么正确、怎么训练、怎么验证」；**不含**服务化架构/并发/存储部署（见 [detailed_design.md](./detailed_design.md)），**不含**实现进度与差距（见 [implementation_status.md](./implementation_status.md) / [gap_analysis.md](./gap_analysis.md)）。

代码载体：`onetrans/models/one_trans.py`、`onetrans/nn/`（blocks/attention/ffn/tokenizer/encoders）、`onetrans/run/`（train/config/builder）、`onetrans/run_baselines/`、`onetrans/serving/two_stage.py`（两阶段引擎）、`onetrans/serving/demo.py`（正确性验证 harness）。

约定符号：`D=d_model`，`H=num_heads`，`d=head_dim=D/H`，`L=num_blocks`，`Ns=ns_tokens_num`（NS token 数），`S_l` 第 `l` 层 S 序列宽度（pyramid 后），`M` 候选数，`T` 目标数（=2）。

---

## 1. 模型总体结构

```
输入特征 ──▶ Embedder（稀疏/多值/分Piecewise编码）──▶ Tokenizer ──▶ OneTrans Backbone ──▶ Head ──▶ logits [B, T]
             YambdaEmbedder                  S / NS 侧             L × CoreOneTransBlock
```

### 1.1 OneTrans Backbone

`L` 个 `CoreOneTransBlock` 堆叠（pre-norm 残差），输入 `(tokens, mask)`，逐层输出裁剪后的 `(tokens', mask')`：

```
h  = RMSNorm(x)
a  = MixedCausalSelfAttention(h, mask)     # 混合参数化注意力（§2）
z  = x + a                                  # 残差 1
z2 = z + MixedFFN(RMSNorm(z))              # 残差 2
# pyramid 降层（§3）：保留尾部 out_seq_num 个 S token + 全部 Ns 个 NS token
```

Head：末 `Ns` 个 token 池化（`x[:, -Ns:, :].mean(1)`）或 CLS token（`x[:, -1, :]`）→ `Linear(D, T)`，输出 `[B, 2]`（is_like / is_full_play 双目标）。

结构超参（默认）：`d_model=256, num_blocks=6, num_heads=4, max_seq_len=100, min_seq_len=10, ns_tokens_num=8`。

### 1.2 Tokenizer

**S 侧（`STokenizer`）**——用户行为序列 → 有序 S token：
1. 各行为类型特征经独立 MLP（`in_dims → D`）；
2. 加 type embedding（第 `i` 类行为 + `type_emb[i]`）；
3. **合并**：`timestamp_aware`（按时间戳跨类型排序，全局时序正确）或 `timestamp_agnostic`（类型拼接 + 可学习分隔 token）。

**NS 侧**——非序列特征（用户画像/候选/上下文）→ Ns 个 NS token：
- `NSGroupWiseTokenizer`：每个特征组一个 MLP，stack 成 `[B, Ns, D]`（组即 token，语义异构来源）；
- `NSAutoSplitTokenizer`：全组拼接后单 MLP 投影到 `l_ns × D`（自动切分）；
- 可选 CLS token（追加一个可学习 token）。

`OneTransTokenizer` 统一封装：S 加位置编码、NS 恒全 1 掩码，输出经 RMSNorm。

**左 padding 语义（全系统正确性根基）**：有效 token 靠序列**尾部**（最新行为在尾），头部为 padding。掩码、有效长度、pyramid 剪裁方向均由此推导。

### 1.3 Embedder（特征表示层）

`YambdaEmbedder`（`onetrans/ext/yambda/` + `nn/encoders/`）：
- **稀疏 ID**：item/user 查 `nn.Embedding`（可选 `MultihashEmbedding` 多哈希降冲突，hash_cardinality=65536, num_hashes=2）；
- **多值特征**（artist_ids/album_ids）：`embedding_bag` mean 池化（多哈希变体拼接）；
- **稠密特征**（15 个 lag 统计量）：`PiecewiseLinearEncoder`（按训练集分位数分桶的线性编码）。

Embedder 属「特征表示」，backbone 属「特征交互」；推理侧稀疏查表由独立 PS 承载（工程见详细设计 §4/§7）。

---

## 2. 混合参数化（Mixed Parametrization）——本模型与普通 Transformer 的本质差异

| token 类别 | attention 投影 | FFN | 物理含义 |
|---|---|---|---|
| S token（行为历史，全部） | 共享 `W_s`（单个 QKV 融合 `Linear(D, 3D)`） | 共享 `network_s` | 历史行为同质，参数共享 |
| NS token（第 i 个） | 独立 `W_ns_list[i]`（各 `Linear(D, 3D)`） | 独立 `networks_ns_list[i]` | 各 NS token 语义异构（画像/候选/上下文），逐 token 参数化提升表达力 |

前向（`MixedCausalSelfAttention.forward`）：S 段一次 `W_s` 投影并 reshape-unbind 出 `(Q_s, K_s, V_s)`；NS 段逐 token `W_ns_list[i]` 投影后 cat；统一转 `bhsd` 做 `scaled_dot_product_attention`，`final_proj` 回投影。`MixedFFN` 同构（S 共享 / NS 逐 token）。

**对推理的深远影响**：S 段共享投影 ⟹ **同一段历史的 K/V 与后段 NS 无关且可预计算**（§4 等价性的来源）；NS 段逐 token 参数化 ⟹ 在线打分时必须逐 token 投影（无法合并为单个 GEMM），这是算子实现的固有形态。

---

## 3. 金字塔降层（Pyramid）

每 block 把 S 宽度从 `dims[l]` 缩到 `dims[l+1]`（`OneTrans.__init__`：`torch.linspace(max_seq_len, min_seq_len, L+1)` 线性，或对数），**保留尾部最新 token**（tail index set `Q = {S_in - out_seq_num+1, …, S_in}`），NS token 恒取末尾 Ns 个不参与缩层。

- **为什么尾部**：配合左 padding，剪裁首先丢弃头部 padding；有效 token 的「最新」语义得以保留——模型越深，感受野越聚焦近期行为。
- **代价**：逐层 `S_l = dims[l]` 不同 ⟹ 缓存的 K/V 各层宽度不同 ⟹ 序列化/掩码必须携带**每层有效长度元数据**（工程契约的核心约束，见详细设计 §4.3/§7.7）。

---

## 4. 两阶段拆分与等价性论证（正确性的根基）

### 4.1 拆分定义

| 维度 | LLM 推理 | 本系统 |
|---|---|---|
| 第一阶段 | prefill（全 prompt 一次编码） | **Nearline（Stage I）**：S 侧逐层编码 + 缓存每层 `(K_s^l, V_s^l)` |
| 第二阶段 | decode（逐 token 自回归） | **Online（Stage II）**：NS 逐层交叉注意力打分（**并行非自回归**） |
| 共享态 | KV cache | **UserKV**（S 侧逐层 K/V） |
| 摊销收益 | 避免重复 prefill | 用户历史编码一次、多次打分摊销 |

**单前向**：`backbone(cat([s_tokens, ns_tokens]))`。
**两阶段**：`encode_s(s)` 只算 S 段（缓存逐层 K/V）→ 存取 → `score_ns(kv, ns)` 每层以 `K=[K_s^l ∥ K_ns^l]` 做注意力。

### 4.2 等价性三不变量

两阶段拼接与单前向**数值等价**，依赖：

- **I1（因果隔离）**：`_s_attn_mask` 上三角因果 ⟹ S 位置只看 S 前缀 ⟹ NS 不进入任何 S 位置的因果窗口 ⟹ S 侧隐藏态与 K/V **与 NS 无关**，可安全预计算；
- **I2（投影一致性）**：`encode_s` 对 S 段用 `W_s`，与单前向对 S 段的投影逐位一致；`score_ns` 对 NS 段用 `W_ns_list[i]`，与单前向对 NS 段一致（共享 `final_proj`/RMSNorm 权重亦一致）；
- **I3（掩码重构一致）**：`score_ns` 从 `per_layer_len[l]` 重构 `s_mask = [zeros(S_l−valid_l) ∥ ones(valid_l)]`，与 `encode_s` 输入掩码在「层 l 输入宽度下的裁剪」语义一致（pyramid 尾部裁剪同步作用于 token 与 mask）。

### 4.3 注意力掩码（左 padding 的正确性核心）

| 掩码 | 形状 | 语义 |
|---|---|---|
| `_s_attn_mask(s_mask)` | `[B,1,S,S]` | S 侧自注意力：列 padding 掩码（无效 key 列 −inf）+ 上三角因果 |
| `_cross_attn_mask(s_mask, ns_len)` | `[B,1,Ns,S+Ns]` | NS→(S∥NS)：S 列只按 padding 掩码（NS 可看全部有效历史）；NS 列按因果掩码（NS 内三角） |

有效长度元数据 `per_layer_len[l]` 是掩码的**唯一依据**——K/V 张量的 `shape[1]` 是 pyramid 满宽 `dims[l]`（恒定），无法从 shape 区分有效 token 数。

### 4.4 两阶段引擎计算逻辑（伪代码）

```python
def encode_s(s_emb, s_mask):                    # Stage I（B=1）
    s, smask, per_layer, plen = s_emb, s_mask, [], []
    s_len = smask.sum()
    for block in blocks:                         # L 层
        plen.append(smask.sum())                 # 该层输入的有效长度（先记）
        h = RMSNorm(s)
        q, k, v = W_s(h).reshape(B,S,3,H,d).unbind(2)
        per_layer.append((k, v))                # ★ 缓存点（I2：与单前向一致）
        z = s + final_proj(SDPA(q,k,v, mask=s_attn_mask(smask)))
        z = z + network_s(RMSNorm(z))
        s = z[:, -out_seq_num:, :]               # pyramid 尾部裁剪（token+mask 同步，I3）
        smask = smask[:, -out_seq_num:, :]
    return UserKV(per_layer, plen, s_len)

def score_ns(kv, ns_emb):                       # Stage II（B=M 候选）
    ns = ns_emb
    for l, block in enumerate(blocks):
        k_s, v_s = kv.per_layer[l]; k_s = k_s.expand(M,...); v_s = v_s.expand(M,...)
        s_mask = cat([zeros(S_l - valid_l), ones(valid_l)])    # 左 padding 重构（I3）
        q,k,v = 逐 token W_ns_list[i](ns[:,i,:]) → cat        # NS 独立投影（I2）
        K, V = cat([k_s, k], dim=1), cat([v_s, v], dim=1)      # 读缓存 + 在线拼接
        z = ns + final_proj(SDPA(q,K,V, mask=cross_attn_mask(s_mask, Ns)))
        z = z + 逐 token networks_ns_list[i](RMSNorm(z))
        ns = z
    return linear(ns[:, -Ns:, :].mean(1))        # [M, T]
```

**批量变体 `score_ns_batch`**：B 个用户的 K/V 逐层 stack（宽度 `dims[l]` 恒定可 stack）、`valid[B] → arange >= S_l−valid` 构造批内左 padding 掩码，与逐条 `score_ns` 数值等价（B=1 时逐位一致）。

### 4.5 与 LLM 推理的关键差异

第二阶段无 sequential 依赖，M 候选整批并行（B 维即 M），瓶颈是「KV 读取带宽 + 交叉注意力算力」而非「逐 token 时延」；故无需 PD 分池，摊销重心是 KV 数据本地性（工程见详细设计 §7）。

---

## 5. 训练设计

### 5.1 数据与样本

- 数据集：Yambda（`onetrans/ext/yambda/`，flat 序列存档）；`DatasetConfig`：multi_event 交互、like 窗口 24h、**lag_seconds=15min**（特征只取 T−15min 之前的历史，防标签泄漏）、按时间切分 train/test（test interval 7 天）、CORE_MIN_INTERACTIONS_PER_ITEM=5 过滤。
- 特征列：15 个稠密 lag 统计（user/item/ui × listen/like/full_play/skip/played_ratio）、多值列（artist_ids/album_ids）、稀疏列（uid/item_id）；标签（is_like, is_full_play）。
- collator（`utils/collator.py`）按 max_seq_len 组装 S 序列与 NS 组，左 padding。

### 5.2 训练循环（`run/train.py`）

- **多目标损失**：`BCEWithLogitsLoss`，`logits [B,2]` vs `labels(is_like, is_full_play)`；
- **混合精度**：`torch.autocast(bfloat16)` + `GradScaler`；
- **指标**：`roc_auc_score`（全局 AUC）、`uauc`（按 user 分组 AUC，更贴排序业务）、`compute_pairwise_accuracy`（同时刻对偏好正确率）；wandb 记录 loss 与累计 FLOPs（`flops_per_sample`，供等算力对比）；
- **checkpoint 生产端**：`save_training_checkpoint` 落盘 embedder/tokenizer/backbone 三段 `state_dict` + meta（model_version/seed）；serving 侧 `weight_loader` 按版本消费（契约见详细设计 §4.4）。

### 5.3 模型构建（`run/builder.py`）

`build_model(archive, …)`：按数据元信息构建 Embedder（item/user/artist/album 嵌入 + Piecewise 稠密编码，可选多哈希）→ S/NS Tokenizer（merge 策略、groupwise/autosplit、CLS 可选）→ OneTrans Backbone。结构与超参由 `run/config.py` 集中管理。

### 5.4 基线（等约束对比，`run_baselines/`）

`run_dcn.py`（DCNv2）、`run_catboost.py`、`run_hiformer.py`（HiFormer）：同 Embedder/同数据/同指标口径，用于「序列 Transformer 精排 vs 传统精排」的 Pareto 对比（算力由累计 FLOPs 对齐）。

---

## 6. 正确性验证设计

### 6.1 验证矩阵

| # | 验证项 | 断言 | 方法 |
|---|---|---|---|
| V1 | 两阶段 vs 单前向等价 | `max|diff| < 1e-4`（fp32 收敛 ≤1e-7 量级） | 固定 seed 构造输入，`demo.py:test_equivalence` |
| V2 | 批量 vs 逐条等价 | `score_ns_batch` 与逐条 `score_ns` 逐位一致 | 同上 `test_dynamic_batching` |
| V3 | 序列化 roundtrip | K/V + 元数据（s_len/per_layer_len）无损往返 | `test_serialize_roundtrip`（含左 padding 用户） |
| V4 | 掩码不变量 | I1/I3：S 段隐藏态不受 NS 影响；重构掩码与原掩码一致 | 构造 NS 扰动对照 + 掩码断言 |
| V5 | 增量 append 语义 | offset/CAS 冲突显式拒绝，绝不静默丢写 | `test_append_conflict` |
| V6 | miss 降级形状 | 缺失用户输出合法形状全零，不抛异常 | `test_kv_miss` |
| V7 | 掩码极端情形 | 全 padding（s_len=0）/ 满长（s_len=S0）用户均可前向 | 边界用例 |

### 6.2 容差与金样本策略

- **分级容差**：fp32 基准 1e-4 起步，收紧至 1e-6；任何 C++ 移植/算子替换必须对同一金样本回放比对（`max|diff|` 不变或收紧）。
- **金样本固化**：固定 seed 输入 + 逐层 K/V 与 logits 快照入库（CI 资产），防止实现漂移。
- **回归纪律**：涉及 `two_stage.py`/`mixed_attention.py`/`serialize.py` 的改动必须全量重跑验证矩阵。

### 6.3 训练侧正确性

- 标签泄漏防护：lag_seconds 截断（§5.1）；
- 时间切分评估：train/test 按时间而非随机切分（贴近线上分布漂移）；
- 指标口径：uAUC/pairwise 与全局 AUC 并报，避免全局 AUC 掩盖个体排序质量。

---

## 7. 与其他文档的对应

| 本文章节 | 对应文档 |
|---|---|
| §1~§3 模型结构 | e2e_design_spec.md §算法边界（概要层） |
| §4 两阶段拆分/等价性 | detailed_design.md §6.1（双轨策略的动因）、§7（serving 工程） |
| §5 训练 | implementation_status.md（训练进度） |
| §6 正确性验证 | detailed_design.md §2 NFR（正确性需求）、implementation_status.md（实测结果） |
