import torch
from torch import nn
from torch.nn.functional import embedding_bag


def _mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=False),
    )


def embed_multivalent(embedding, values, lengths):
    batch_size = lengths.shape[0]
    if values.dim() == 1:
        values = values.unsqueeze(1)
    num_hashes = values.shape[1]
    offsets = torch.zeros(batch_size, dtype=torch.long, device=values.device)
    offsets[1:] = lengths.cumsum(dim=0)[:-1]
    outputs = []
    for h in range(num_hashes):
        emb = embedding_bag(
            values[:, h].contiguous(),
            embedding.weight,
            offsets=offsets,
            mode="mean",
            sparse=False,
        )
        outputs.append(emb)
    return torch.cat(outputs, dim=1)


class STokenizer(nn.Module):
    def __init__(self, d_model, in_dims, merge="timestamp_aware"):
        super().__init__()
        self.n_seq_types = len(in_dims)
        self.merge = merge

        self.mlps = nn.ModuleList([_mlp(dim, d_model) for dim in in_dims])
        self.type_embeddings = nn.Embedding(self.n_seq_types, d_model)

        if merge == "timestamp_agnostic":
            self.sep_tokens = nn.Parameter(torch.zeros(self.n_seq_types - 1, d_model))

    def _timestamp_aware_merge(self, tokens_list, masks, timestamps):
        for i in range(self.n_seq_types):
            tokens_list[i] = tokens_list[i] + self.type_embeddings.weight[i]

        all_tokens = torch.cat(tokens_list, dim=1)
        all_ts = torch.cat(timestamps, dim=1)
        all_masks = torch.cat(masks, dim=1)

        sort_ts = all_ts.masked_fill(~all_masks, all_ts.max() + 1)
        order = sort_ts.argsort(dim=1)

        sorted_tokens = all_tokens.gather(1, order.unsqueeze(-1).expand_as(all_tokens))
        sorted_mask = all_masks.gather(1, order)
        sorted_tokens = sorted_tokens * sorted_mask.unsqueeze(-1)
        return sorted_tokens, sorted_mask

    def _timestamp_agnostic_merge(self, tokens_list, masks):
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        parts_t, parts_m = [], []
        for i, (tokens, mask) in enumerate(zip(tokens_list, masks)):
            parts_t.append(tokens)
            parts_m.append(mask)
            if i < self.n_seq_types - 1:
                sep = self.sep_tokens[i].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
                parts_t.append(sep)
                parts_m.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        all_tokens = torch.cat(parts_t, dim=1)
        all_masks = torch.cat(parts_m, dim=1)
        return all_tokens * all_masks.unsqueeze(-1), all_masks

    def forward(self, sequences_features, sequences_masks, sequences_timestamps=None):
        tokens_list = [
            self.mlps[i](sequences_features[i]) * sequences_masks[i].unsqueeze(-1)
            for i in range(self.n_seq_types)
        ]

        if self.merge == "timestamp_aware":
            return self._timestamp_aware_merge(tokens_list, sequences_masks, sequences_timestamps)
        else:
            return self._timestamp_agnostic_merge(tokens_list, sequences_masks)


class NSGroupWiseTokenizer(nn.Module):
    def __init__(self, d_model, in_dims):
        super().__init__()
        self.mlps = nn.ModuleList([_mlp(dim, d_model) for dim in in_dims])

    @property
    def n_ns_tokens(self):
        return len(self.mlps)

    def forward(self, groups):
        return torch.stack([mlp(g) for mlp, g in zip(self.mlps, groups)], dim=1)


class NSAutoSplitTokenizer(nn.Module):
    def __init__(self, d_model, l_ns, in_dims):
        super().__init__()
        self._l_ns = l_ns
        self._d_model = d_model
        self.proj = _mlp(sum(in_dims), d_model * l_ns)

    @property
    def n_ns_tokens(self):
        return self._l_ns

    def forward(self, groups):
        ns_concat = torch.cat(groups, dim=1)
        projected = self.proj(ns_concat)
        return projected.reshape(projected.shape[0], self._l_ns, self._d_model)


class OneTransTokenizer(nn.Module):
    def __init__(self, s_tokenizer, ns_tokenizer, d_model, max_seq_len, use_cls_token=False):
        super().__init__()
        self.s_tokenizer = s_tokenizer
        self.ns_tokenizer = ns_tokenizer
        self.rms_norm = nn.RMSNorm(d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @property
    def n_ns_tokens(self):
        return self.ns_tokenizer.n_ns_tokens + (1 if self.use_cls_token else 0)

    def encode_s(self, seq_features, seq_masks, seq_timestamps=None):
        """仅编码 S（用户历史）侧，返回 RMSNorm 后的 S token 与掩码（nearline/Stage I）。"""
        s_tokens, s_mask = self.s_tokenizer(seq_features, seq_masks, seq_timestamps)
        B, L, _ = s_tokens.shape
        positions = torch.arange(L, device=s_tokens.device).unsqueeze(0)
        s_tokens = s_tokens + self.pos_embedding(positions)
        return self.rms_norm(s_tokens), s_mask

    def encode_ns(self, ns_groups):
        """仅编码 NS（非序列/候选）侧，返回 RMSNorm 后的 NS token 与掩码（online/Stage II）。"""
        ns_tokens = self.ns_tokenizer(ns_groups)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(ns_tokens.shape[0], 1, -1)
            ns_tokens = torch.cat([ns_tokens, cls_tokens], dim=1)
        L_NS = ns_tokens.shape[1]
        ns_mask = torch.ones(ns_tokens.shape[0], L_NS, dtype=torch.bool, device=ns_tokens.device)
        return self.rms_norm(ns_tokens), ns_mask

    def forward(self, batch):
        s_tokens, s_mask = self.encode_s(
            batch["seq_features"],
            batch["seq_masks"],
            batch.get("seq_timestamps"),
        )
        ns_tokens, ns_mask = self.encode_ns(batch["ns_groups"])

        tokens = torch.cat([s_tokens, ns_tokens], dim=1)
        mask = torch.cat([s_mask, ns_mask], dim=1)

        return tokens, mask
