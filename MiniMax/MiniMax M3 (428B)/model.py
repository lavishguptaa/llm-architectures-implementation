import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(rotary_dim, theta_base, context_length):

    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # Partial RoPE: only the first rotary_dim dims of each head are rotated,
    # the remaining dims pass through unchanged.
    batch_size, num_heads, seq_len, head_dim = x.shape
    rotary_dim = cos.shape[-1]

    cos = cos[:seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[:seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., :rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    x_rot = (x_rot * cos) + (rotated * sin)

    return torch.cat([x_rot, x_pass], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x_float = x.float()

        x_norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return (x_norm * (1.0 + self.weight.float())).to(x.dtype)


class BlockIndexer(nn.Module):
    # Scores every key for every query with a small multi-head dot-product, max-pools the scores
    # inside blocks of block_size keys, and keeps the top-k key blocks per query (plus the
    # query's own recent blocks, which are always kept). Only the selection is computed here:
    # there is no value branch.

    def __init__(self, cfg, sparse_cfg):
        super().__init__()

        if sparse_cfg["sparse_score_type"] != "max":
            raise ValueError("only the 'max' sparse_score_type is supported")

        if sparse_cfg["sparse_init_block"] != 0:
            raise ValueError("sparse_init_block must be 0")

        self.head_dim = sparse_cfg["sparse_index_dim"]
        self.num_heads = sparse_cfg["sparse_num_index_heads"]
        self.block_size = sparse_cfg["sparse_block_size"]
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.local_blocks = sparse_cfg["sparse_local_block"]

        # One index head per key/value head, so every GQA group makes its own block selection
        if self.num_heads != cfg["num_key_value_heads"]:
            raise ValueError("the number of index heads must equal num_key_value_heads")

        d_in = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]

        self.q_proj = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_in, self.head_dim, bias=False)  # a single shared key head
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def forward(self, x, cos, sin):
        batch_size, num_tokens, _ = x.shape

        q = self.q_proj(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_tokens, 1, self.head_dim)

        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)

        positions = torch.arange(num_tokens, device=x.device)
        future = positions[None, :] > positions[:, None]          # [S_q, S_k]

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2))   # [B, H_idx, S_q, S_k]
        scores = scores.masked_fill(future, float("-inf"))

        # Max-pool the key scores inside each block
        num_blocks = -(-num_tokens // self.block_size)
        pad = num_blocks * self.block_size - num_tokens

        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))

        block_scores = scores.view(
            batch_size, self.num_heads, num_tokens, num_blocks, self.block_size
        ).amax(dim=-1)                                            # [B, H_idx, S_q, num_blocks]

        # The query's own block and the (local_blocks - 1) before it always win a slot
        q_block = positions // self.block_size
        local = torch.arange(self.local_blocks, device=x.device)
        local_idx = (q_block[:, None] - local[None, :]).clamp(min=0)       # [S_q, local]
        local_idx = local_idx[None, None].expand(batch_size, self.num_heads, -1, -1)
        block_scores.scatter_(-1, local_idx, float("inf"))

        topk = min(self.topk_blocks, num_blocks)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)

        # Slots that fall on a future block (score -inf) are unused
        return topk_indices.masked_fill(topk_scores == float("-inf"), -1)

    def build_mask(self, block_indices, num_tokens, num_query_heads):
        # Expand the chosen blocks to a boolean [B, num_query_heads, S_q, S_k] mask (True = may attend)
        batch_size, num_index_heads, _, _ = block_indices.shape
        num_blocks = -(-num_tokens // self.block_size)

        # -1 (unused) slots land in a throwaway extra column that is dropped again
        safe = block_indices.masked_fill(block_indices < 0, num_blocks)
        keep = torch.zeros(
            batch_size, num_index_heads, num_tokens, num_blocks + 1,
            dtype=torch.bool, device=block_indices.device,
        )
        keep.scatter_(-1, safe, True)
        keep = keep[..., :num_blocks]

        # Back from blocks to individual keys, then from index heads to query heads
        keep = keep.repeat_interleave(self.block_size, dim=-1)[..., :num_tokens]
        keep = keep.repeat_interleave(num_query_heads // num_index_heads, dim=1)

        positions = torch.arange(num_tokens, device=block_indices.device)
        causal = positions[None, :] <= positions[:, None]

        return keep & causal


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, use_sparse):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg.get("attention_dropout", 0.0)

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=False)

        # Per-head QK-norm (Gemma-style), applied before RoPE
        self.q_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])

        # Sparse layers only attend to the key blocks picked by the indexer
        self.indexer = BlockIndexer(cfg, cfg["sparse_attention_config"]) if use_sparse else None

    def forward(self, x, cos, sin):
        batch_size, num_tokens, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)

        Q = self.q_norm(Q).transpose(1, 2)
        K = self.k_norm(K).transpose(1, 2)
        V = V.transpose(1, 2)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        attn_mask = None
        is_causal = True

        if self.indexer is not None:
            block_indices = self.indexer(x, cos, sin)
            attn_mask = self.indexer.build_mask(block_indices, num_tokens, self.num_heads)
            is_causal = False

        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)


class FeedForward(nn.Module):
    # Clamped SwiGLU ("swigluoai"): used for the dense layers, the shared expert and every routed expert
    def __init__(self, emb_dim, hidden_dim, alpha, limit):
        super().__init__()

        self.alpha = alpha
        self.limit = limit

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)

        glu = gate * torch.sigmoid(gate * self.alpha)

        return self.down_proj((up + 1.0) * glu)


class MoEFeedForward(nn.Module):
    # Sigmoid-routed experts plus a shared expert that every token uses

    def __init__(self, cfg):
        super().__init__()

        if cfg["scoring_func"] != "sigmoid":
            raise ValueError("only the sigmoid scoring_func is supported")

        if not cfg["use_routing_bias"]:
            raise ValueError("use_routing_bias must be true")

        if cfg["n_shared_experts"] != 1:
            raise ValueError("exactly one shared expert is supported")

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_local_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")

        alpha, limit = cfg["swiglu_alpha"], cfg["swiglu_limit"]

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        # Bias used only to pick the experts, never to weight them
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["intermediate_size"], alpha, limit)
                for _ in range(self.num_experts)
            ]
        )

        self.shared_experts = FeedForward(self.emb_dim, cfg["shared_intermediate_size"], alpha, limit)

    def route(self, x_flat):
        # Sigmoid scores (not softmax) over the experts
        scores = torch.sigmoid(self.gate(x_flat).float())

        topk_indices = torch.topk(scores + self.e_score_correction_bias, self.num_experts_per_tok, dim=-1)[1]

        # Mixing weights come from the original scores, renormalised over the chosen experts
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_indices

    def forward(self, x):
        batch, seq_len, _ = x.shape

        x_flat = x.reshape(batch * seq_len, self.emb_dim)

        topk_weights, topk_indices = self.route(x_flat)
        topk_weights = topk_weights.to(x_flat.dtype)

        routed_out = torch.zeros_like(x_flat)

        # Run only the experts that received tokens
        for expert_id in torch.unique(topk_indices).tolist():
            token_idx, slot_idx = torch.where(topk_indices == expert_id)

            expert_out = self.experts[expert_id](x_flat[token_idx])
            expert_out = expert_out * topk_weights[token_idx, slot_idx].unsqueeze(-1)

            routed_out.index_add_(0, token_idx, expert_out)

        # Only the routed experts are scaled; the shared expert is added as is
        output = routed_out * self.routed_scaling_factor + self.shared_experts(x_flat)

        return output.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        sparse_attention = bool(cfg["sparse_attention_config"]["sparse_attention_freq"][layer_idx])
        use_moe = bool(cfg["moe_layer_freq"][layer_idx])

        self.attn = GroupedQueryAttention(cfg, sparse_attention)

        if use_moe:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(
                cfg["hidden_size"], cfg["dense_intermediate_size"],
                cfg["swiglu_alpha"], cfg["swiglu_limit"],
            )

        self.layer1 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.layer2 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, cos, sin):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x


class Model(nn.Module):
    # Text decoder only. The vision tower, the image projector and the multi-token-prediction
    # modules (num_mtp_modules; the released checkpoint has no weights for them) are not implemented.

    def __init__(self, cfg):
        super().__init__()

        if not cfg["use_gemma_norm"]:
            raise ValueError("only Gemma-style RMSNorm is supported")

        if cfg["attention_output_gate"]:
            raise ValueError("attention_output_gate is not supported")

        if not cfg["use_qk_norm"] or cfg["qk_norm_type"] != "per_head":
            raise ValueError("only per-head QK-norm is supported")

        if not cfg["sparse_attention_config"]["use_sparse_attention"]:
            raise ValueError("this implementation expects sparse attention")

        if len(cfg["moe_layer_freq"]) != cfg["num_hidden_layers"]:
            raise ValueError("moe_layer_freq length must equal num_hidden_layers")

        if len(cfg["sparse_attention_config"]["sparse_attention_freq"]) != cfg["num_hidden_layers"]:
            raise ValueError("sparse_attention_freq length must equal num_hidden_layers")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # Only the first rotary_dim dimensions of every head are rotated (64 of 128).
        # One RoPE table is shared by every layer, and by the indexer.
        cos, sin = compute_rope_params(
            cfg["rotary_dim"], cfg["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        if idx.size(1) > self.cos.size(0):
            raise ValueError(
                f"Sequence length {idx.size(1)} exceeds "
                f"max_position_embeddings {self.cos.size(0)}"
            )

        x = self.tok_emb(idx)

        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        x = self.out_norm_layer(x)
        return self.out_proj(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():

    with open("config.json") as f:
        cfg = json.load(f)

    # config.json is the multimodal config; the language model's settings are under "text_config"
    cfg = cfg["text_config"]

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()

    input_text = "Hello, how are you?"

    tokenizer = Tokenizer.from_file("tokenizer.json")

    input_ids = tokenizer.encode(input_text).ids
    encoded_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=cfg["max_position_embeddings"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
