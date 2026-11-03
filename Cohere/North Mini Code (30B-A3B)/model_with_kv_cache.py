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

    # Interleaved layout: every frequency is repeated for two neighbouring dims (GPT-J style),
    # [f0, f0, f1, f1, ...] rather than [f0, f1, ..., f0, f1, ...]
    angles = torch.repeat_interleave(angles, 2, dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin, start_pos=0):

    batch_size, num_heads, seq_len, head_dim = x.shape

    cos = cos[start_pos:start_pos + seq_len, :].to(x.device)
    sin = sin[start_pos:start_pos + seq_len, :].to(x.device)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_float = x.float()

    # Rotate neighbouring pairs (x0, x1) -> (-x1, x0)
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    return ((x_float * cos) + (rotated * sin)).to(x.dtype)


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


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        x_float = x.float()

        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).pow(2).mean(dim=-1, keepdim=True)

        x_norm = (x_float - mean) * torch.rsqrt(var + self.eps)

        return (self.weight.float() * x_norm).to(x.dtype)


def build_norm(cfg):
    # RMSNorm when rms_norm_eps is set, otherwise bias-free LayerNorm with layer_norm_eps
    if cfg.get("rms_norm_eps") is not None:
        return RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    return LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx, use_rope):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.attention_type = cfg["layer_types"][layer_idx]

        if self.attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]

        if cfg.get("use_qk_norm", False):
            raise ValueError("use_qk_norm is not supported")

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        # Sliding-window layers use RoPE. Full-attention layers have no positional encoding (NoPE),
        # except the dense prefix layers, which are given RoPE.
        self.use_rope = use_rope
        self.sliding_window = cfg["sliding_window"] if self.attention_type == "sliding_attention" else None

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

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

        if self.use_rope:
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

        if self.sliding_window is not None:

            q_pos = torch.arange(past_len, past_len + num_tokens, device=x.device)[:, None]
            k_pos = torch.arange(past_len + num_tokens, device=x.device)[None, :]

            attn_mask = (
                (k_pos <= q_pos)
                & (k_pos > q_pos - self.sliding_window)
            )
            is_causal = False

        elif past_len > 0:
            is_causal = False

            # A single new token may attend to everything; several new tokens need a mask
            # that is offset by the number of cached tokens
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
    # SwiGLU. Used for the dense prefix layer, the shared experts and every routed expert.
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
    # Routed experts plus an optional shared expert block, combined by averaging

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.expert_selection_fn = cfg["expert_selection_fn"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.num_shared_experts = cfg["num_shared_experts"]
        self.shared_combination = cfg["shared_expert_combination_strategy"]

        if self.expert_selection_fn not in ("softmax", "sigmoid"):
            raise ValueError("expert_selection_fn must be 'softmax' or 'sigmoid'")

        if self.shared_combination not in ("sum", "average"):
            raise ValueError("shared_expert_combination_strategy must be 'sum' or 'average'")

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        # No selection bias: experts are picked directly from the router logits
        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["intermediate_size"])
                for _ in range(self.num_experts)
            ]
        )

        # The shared experts are fused into one wide FFN
        if self.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                self.emb_dim, cfg["intermediate_size"] * self.num_shared_experts
            )

    def route(self, x_flat):
        logits = self.gate(x_flat)

        # Pick the top-k logits first, then turn only those into mixing weights
        topk_logits, topk_indices = torch.topk(logits, self.num_experts_per_tok, dim=-1)

        if self.expert_selection_fn == "softmax":
            topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float)
        else:
            topk_weights = torch.sigmoid(topk_logits)

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

        if self.num_shared_experts > 0:
            shared_out = self.shared_experts(x_flat)

            if self.shared_combination == "average":
                routed_out = (routed_out + shared_out) / 2
            else:
                routed_out = routed_out + shared_out

        return routed_out.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    # Parallel block: attention and the feed-forward read the same normalised input,
    # and both outputs are added to the residual stream.
    def __init__(self, cfg, layer_idx):
        super().__init__()

        is_dense = cfg["mlp_layer_types"][layer_idx] == "dense"

        use_rope = (
            cfg["layer_types"][layer_idx] == "sliding_attention"
            or (is_dense and cfg["prefix_dense_sliding_window_pattern"] == 1)
        )

        self.attn = GroupedQueryAttention(cfg, layer_idx, use_rope)

        if is_dense:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["prefix_dense_intermediate_size"])
        else:
            self.ffn = MoEFeedForward(cfg)

        self.layer1 = build_norm(cfg)

    def forward(self, x, cos, sin, layer_cache=None):
        shortcut = x
        x = self.layer1(x)

        # Parallel block: attention and the feed-forward read the same normalised input
        attn_out, new_cache = self.attn(x, cos, sin, layer_cache)

        return shortcut + attn_out + self.ffn(x), new_cache


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if cfg["num_hidden_layers"] != len(cfg["layer_types"]):
            raise ValueError("layer_types length must equal num_hidden_layers")

        if cfg["rope_scaling"] is not None:
            raise ValueError("rope_scaling is not supported")

        # Layers before first_k_dense_replace use a dense FFN instead of the MoE
        if "mlp_layer_types" not in cfg:
            first_k_dense = cfg.get("first_k_dense_replace", 0)
            cfg = dict(
                cfg,
                mlp_layer_types=[
                    "dense" if i < first_k_dense else "sparse"
                    for i in range(cfg["num_hidden_layers"])
                ],
            )

        self.logit_scale = cfg["logit_scale"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = build_norm(cfg)
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        # config.json does not list tie_word_embeddings; the Cohere2 MoE default is True
        if cfg.get("tie_word_embeddings", True):
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer that uses RoPE
        cos, sin = compute_rope_params(
            cfg["head_dim"], cfg["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, cache=None, use_cache=False):
        # cache: list with one entry per layer, from the previous forward call
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
        logits = self.out_proj(x) * self.logit_scale

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

    # Some releases nest the language-model settings under "text_config"
    cfg = cfg.get("text_config", cfg)

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
