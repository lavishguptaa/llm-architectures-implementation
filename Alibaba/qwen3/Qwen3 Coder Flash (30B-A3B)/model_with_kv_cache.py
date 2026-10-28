import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(head_dim, theta_base, context_length):

    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin, start_pos=0):

    batch_size, num_heads, seq_len, head_dim = x.shape

    cos = cos[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        x_float = x.float()

        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        return self.weight * (x_float / rms).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

        # Per-head RMSNorm on queries and keys, applied before RoPE
        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, layer_cache=None):
        # layer_cache: (K, V) from earlier steps, each [batch, kv_heads, past_len, head_dim].
        # Only the key/value heads are cached.
        batch_size, num_tokens, _ = x.shape

        past_len = 0 if layer_cache is None else layer_cache[0].size(2)

        Q = self.W_q(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.W_k(x).view(
            batch_size, num_tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        V = self.W_v(x).view(
            batch_size, num_tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = apply_rope(Q, cos, sin, start_pos=past_len)
        K = apply_rope(K, cos, sin, start_pos=past_len)

        # Keys are cached after RoPE, so they are never recomputed
        if layer_cache is not None:
            K = torch.cat([layer_cache[0], K], dim=2)
            V = torch.cat([layer_cache[1], V], dim=2)

        new_cache = (K, V)

        # With a cache the queries are only the new tokens but the keys cover the whole
        # sequence, so the built-in causal mask (aligned to the top-left) would be wrong.
        attn_mask = None
        is_causal = True

        if past_len > 0:
            is_causal = False

            if num_tokens > 1:
                q_pos = torch.arange(past_len, past_len + num_tokens, device=x.device)[:, None]
                k_pos = torch.arange(past_len + num_tokens, device=x.device)[None, :]
                attn_mask = k_pos <= q_pos

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

        return self.out_proj(context), new_cache


class FeedForward(nn.Module):
    # SwiGLU. Used for the dense layers and as every routed expert.
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class MoEFeedForward(nn.Module):
    # Sparse mixture of experts: no shared expert, softmax router over all experts

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.norm_topk_prob = cfg["norm_topk_prob"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["moe_intermediate_size"])
                for _ in range(self.num_experts)
            ]
        )

    def route(self, x_flat):
        # Softmax over every expert, then keep the top-k probabilities
        probs = torch.softmax(self.gate(x_flat).float(), dim=-1)

        topk_weights, topk_indices = torch.topk(probs, self.num_experts_per_tok, dim=-1)

        if self.norm_topk_prob:
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

        return routed_out.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = GroupedQueryAttention(cfg)

        # A layer is sparse unless it is listed in mlp_only_layers; decoder_sparse_step
        # makes only every n-th layer sparse
        is_sparse = (
            layer_idx not in cfg["mlp_only_layers"]
            and cfg["num_experts"] > 0
            and (layer_idx + 1) % cfg["decoder_sparse_step"] == 0
        )

        if is_sparse:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])

        self.layer1 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.layer2 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, layer_cache=None):
        shortcut = x
        x = self.layer1(x)
        x, new_cache = self.attn(x, cos, sin, layer_cache)
        x = x + shortcut

        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, new_cache


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if cfg["rope_scaling"] is not None:
            raise ValueError("rope_scaling is not supported")

        if cfg["use_sliding_window"]:
            raise ValueError("sliding window attention is not supported")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer
        cos, sin = compute_rope_params(
            cfg["head_dim"], cfg["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, cache=None, use_cache=False):
        # cache: list with one (K, V) pair per layer, from the previous forward call
        # use_cache=False returns logits only; use_cache=True returns (logits, cache)
        past_len = 0 if cache is None else cache[0][0].size(2)

        if past_len + idx.size(1) > self.cos.size(0):
            raise ValueError(
                f"Sequence length {past_len + idx.size(1)} exceeds "
                f"max_position_embeddings {self.cos.size(0)}"
            )

        x = self.tok_emb(idx)

        new_cache = []

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[layer_idx]
            x, layer_cache = block(x, self.cos, self.sin, layer_cache)
            new_cache.append(layer_cache)

        x = self.out_norm_layer(x)
        logits = self.out_proj(x)

        if use_cache:
            return logits, new_cache
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):

    if idx.size(1) + max_new_tokens > context_size:
        raise ValueError(
            "prompt length + max_new_tokens must not exceed context_size "
            "when using the KV cache"
        )

    with torch.no_grad():
        # Prefill: process the whole prompt once and fill the cache
        logits, cache = model(idx, use_cache=True)

        for step in range(max_new_tokens):

            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

            # Decode: feed only the new token; earlier K/V come from the cache
            if step < max_new_tokens - 1:
                logits, cache = model(idx_next, cache=cache, use_cache=True)

    return idx


def main():

    with open("config.json") as f:
        cfg = json.load(f)

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
