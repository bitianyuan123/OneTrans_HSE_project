import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

# ----------------------------------------------------------------------
# Configuration (adjust as needed)
# ----------------------------------------------------------------------
class Config:
    d_model: int = 256          # token dimension D
    num_tokens: int = 8         # T (must divide d_model if H=T)
    num_layers: int = 2
    expansion_ratio: int = 4    # k for FFN hidden dim expansion
    num_tasks: int = 2          # e.g., is_like, is_full_play
    # sequence module
    seq_max_len: int = 50
    # dense features dimension (set according to your data)
    num_dense_features: int = 128   # update with actual number of DENSE_COLUMNS


# ----------------------------------------------------------------------
# Helper: Target Attention for sequence features
# ----------------------------------------------------------------------
class TargetAttention(nn.Module):
    """DIN‑style target attention: query = target item, keys = history items."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, query: torch.Tensor, keys: torch.Tensor, lengths: torch.Tensor):
        """
        query: (B, D)
        keys:  (B, L, D)   (padded sequence)
        lengths: (B,)      actual lengths
        returns: (B, D)
        """
        # Scaled dot‑product attention (no learnable parameters)
        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)).squeeze(1)  # (B, L)
        scores = scores / (self.d_model ** 0.5)

        # Mask padding positions
        mask = torch.arange(keys.size(1), device=keys.device) < lengths.unsqueeze(1)  # (B, L)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(1)      # (B, 1, L)

        context = torch.matmul(attn_weights, keys).squeeze(1)  # (B, D)
        return context


# ----------------------------------------------------------------------
# Embedder: processes S and NS parts of the batch and returns a concatenated vector
# ----------------------------------------------------------------------
class RankMixerEmbedder(nn.Module):
    def __init__(self, config: Config, meta: Dict):
        super().__init__()
        self.config = config

        # Embedding tables
        self.user_emb = nn.Embedding(meta['num_users'] + 1, config.d_model, padding_idx=0)
        self.item_emb = nn.Embedding(meta['num_items'] + 1, config.d_model, padding_idx=0)
        self.artist_emb = nn.Embedding(meta['num_artists'] + 1, config.d_model, padding_idx=0)
        self.album_emb = nn.Embedding(meta['num_albums'] + 1, config.d_model, padding_idx=0)

        # Dense features -> MLP
        self.dense_mlp = nn.Sequential(
            nn.Linear(config.num_dense_features, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Sequence module
        self.target_attn = TargetAttention(config.d_model)

    def forward(self, batch: Dict) -> torch.Tensor:
        # ----- NS (non‑sequential) features -----
        # uid, target item_id
        uid = batch['NS']['sparse_features']['uid']                 # (B,)
        target_item_id = batch['NS']['sparse_features']['item_id']  # (B,)

        user_emb = self.user_emb(uid)                    # (B, D)
        target_item_emb = self.item_emb(target_item_id)  # (B, D)

        # dense features
        dense = batch['NS']['dense_features']            # (B, num_dense_features)
        dense_emb = self.dense_mlp(dense)                # (B, D)

        # multivalent features: artist_ids (NS)
        artist_ids_ns = batch['NS']['multivalent_features']['artist_ids']['values']   # (B, max_artist_len)
        artist_len = batch['NS']['multivalent_features']['artist_ids']['length']      # (B,)
        artist_emb_ns = self.artist_emb(artist_ids_ns)                                # (B, max_len, D)
        # mask and mean pool
        artist_mask = torch.arange(artist_ids_ns.size(1), device=artist_ids_ns.device) < artist_len.unsqueeze(1)
        artist_emb_ns = (artist_emb_ns * artist_mask.unsqueeze(-1)).sum(dim=1) / artist_len.float().unsqueeze(-1)   # (B, D)

        # multivalent features: album_ids (NS)
        album_ids_ns = batch['NS']['multivalent_features']['album_ids']['values']     # (B, max_album_len)
        album_len = batch['NS']['multivalent_features']['album_ids']['length']        # (B,)
        album_emb_ns = self.album_emb(album_ids_ns)                                   # (B, max_len, D)
        album_mask = torch.arange(album_ids_ns.size(1), device=album_ids_ns.device) < album_len.unsqueeze(1)
        album_emb_ns = (album_emb_ns * album_mask.unsqueeze(-1)).sum(dim=1) / album_len.float().unsqueeze(-1)   # (B, D)

        # ----- S (sequential) features -----
        seq_item_ids = batch['S']['item_id']        # (B, seq_len)
        seq_lengths = batch['S']['length']          # (B,)
        seq_item_embs = self.item_emb(seq_item_ids) # (B, seq_len, D)
        seq_context = self.target_attn(target_item_emb, seq_item_embs, seq_lengths)  # (B, D)

        # optional cross feature (user * item)
        cross_feat = user_emb * target_item_emb     # (B, D)

        # Concatenate all field embeddings -> one vector per sample
        concat_vec = torch.cat([
            user_emb,
            target_item_emb,
            dense_emb,
            artist_emb_ns,
            album_emb_ns,
            seq_context,
            cross_feat
        ], dim=-1)   # (B, total_dim = 7 * D)
        return concat_vec


# ----------------------------------------------------------------------
# Semantic Tokenization: split concatenated vector into T tokens of dimension D
# ----------------------------------------------------------------------
class SemanticTokenizer(nn.Module):
    def __init__(self, total_dim: int, num_tokens: int, token_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        chunk_size = total_dim // num_tokens
        # If total_dim is not divisible by num_tokens, we can pad or adapt.
        # Here we assume divisibility (total_dim = 7*D and D is chosen appropriately).
        self.chunk_size = chunk_size
        self.projections = nn.ModuleList([
            nn.Linear(chunk_size, token_dim) for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, total_dim)
        chunks = torch.split(x, self.chunk_size, dim=-1)   # list of (B, chunk_size)
        tokens = []
        for i, chunk in enumerate(chunks):
            tok = self.projections[i](chunk)              # (B, token_dim)
            tokens.append(tok.unsqueeze(1))
        return torch.cat(tokens, dim=1)                   # (B, num_tokens, token_dim)


# ----------------------------------------------------------------------
# RankMixer Block components
# ----------------------------------------------------------------------
class TokenMixer(nn.Module):
    """Parameter‑free multi‑head token mixing (H = T)."""
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        assert dim % num_tokens == 0, f"dim {dim} must be divisible by num_tokens {num_tokens}"
        self.num_tokens = num_tokens
        self.head_dim = dim // num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # (B, T, H, head_dim) with H = T
        x = x.view(B, T, self.num_tokens, self.head_dim)
        # (B, H, T, head_dim)
        x = x.transpose(1, 2)
        # Flatten last two dims -> (B, H, T*head_dim) = (B, T, D)
        x = x.reshape(B, self.num_tokens, -1)
        return x


class PerTokenFFN(nn.Module):
    """Separate FFN for each token."""
    def __init__(self, num_tokens: int, dim: int, expansion_ratio: int):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        outputs = []
        for i in range(x.size(1)):
            token_out = self.experts[i](x[:, i, :])   # (B, D)
            outputs.append(token_out.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class RankMixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int, expansion_ratio: int):
        super().__init__()
        self.token_mixer = TokenMixer(num_tokens, dim)
        self.ffn = PerTokenFFN(num_tokens, dim, expansion_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(x)
        x = self.norm1(x + mixed)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ----------------------------------------------------------------------
# Full RankMixer model
# ----------------------------------------------------------------------
class RankMixer(nn.Module):
    def __init__(self, config: Config, meta: Dict):
        super().__init__()
        self.config = config
        self.embedder = RankMixerEmbedder(config, meta)

        # total dimension after concatenation = 7 * d_model
        total_dim = 7 * config.d_model
        self.tokenizer = SemanticTokenizer(total_dim, config.num_tokens, config.d_model)

        self.blocks = nn.ModuleList([
            RankMixerBlock(config.num_tokens, config.d_model, config.expansion_ratio)
            for _ in range(config.num_layers)
        ])

        self.head = nn.Linear(config.d_model, config.num_tasks)

    def forward(self, batch: Dict) -> torch.Tensor:
        concat_vec = self.embedder(batch)          # (B, total_dim)
        tokens = self.tokenizer(concat_vec)        # (B, T, D)
        for block in self.blocks:
            tokens = block(tokens)                 # (B, T, D)
        pooled = tokens.mean(dim=1)                # (B, D)
        logits = self.head(pooled)                 # (B, num_tasks)
        return logits


# ----------------------------------------------------------------------
# Example usage with dummy data (to test shapes)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Mock meta information (vocab sizes)
    meta = {
        'num_users': 10000,
        'num_items': 50000,
        'num_artists': 5000,
        'num_albums': 8000,
    }
    cfg = Config()
    cfg.num_dense_features = 128   # adjust to your DENSE_COLUMNS length
    cfg.d_model = 256
    cfg.num_tokens = 8             # T
    cfg.expansion_ratio = 4
    cfg.num_layers = 2

    model = RankMixer(cfg, meta)

    # Create a dummy batch (batch size = 4)
    B = 4
    seq_len = 30   # max sequence length after padding
    batch = {
        "S": {
            "item_id": torch.randint(1, meta['num_items'], (B, seq_len)),
            "length": torch.tensor([20, 25, 18, 30]),  # actual lengths
        },
        "NS": {
            "sparse_features": {
                "uid": torch.randint(1, meta['num_users'], (B,)),
                "item_id": torch.randint(1, meta['num_items'], (B,)),
            },
            "dense_features": torch.randn(B, cfg.num_dense_features),
            "multivalent_features": {
                "artist_ids": {
                    "values": torch.randint(1, meta['num_artists'], (B, 5)),   # max 5 artists
                    "length": torch.tensor([3, 4, 2, 5]),
                },
                "album_ids": {
                    "values": torch.randint(1, meta['num_albums'], (B, 3)),    # max 3 albums
                    "length": torch.tensor([2, 1, 3, 2]),
                }
            }
        },
        "targets": {
            "is_like": torch.randint(0, 2, (B,)),
            "is_full_play": torch.randint(0, 2, (B,))
        }
    }

    logits = model(batch)
    print("Logits shape:", logits.shape)   # expected (B, num_tasks) = (4, 2)