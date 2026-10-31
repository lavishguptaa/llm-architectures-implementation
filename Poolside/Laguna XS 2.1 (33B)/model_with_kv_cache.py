import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(head_dim, rope_cfg, context_length):
    # rope_cfg is the rope_parameters entry of one layer type: "default" or "yarn"

    partial_rotary_factor = rope_cfg.get("partial_rotary_factor", 1.0)
    theta_base = rope_cfg["rope_theta"]
    rotary_dim = int(head_dim * partial_rotary_factor)

    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    attention_factor = 1.0

    pos_freqs = theta_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)

    inv_freq = 1.0 / pos_freqs

    if rope_cfg["rope_type"] == "yarn":
        factor = rope_cfg["factor"]
        orig_max = rope_cfg["original_max_position_embeddings"]
        beta_fast = rope_cfg["beta_fast"]
        beta_slow = rope_cfg["beta_slow"]
        attention_factor = rope_cfg["attention_factor"]

        def find_correction_dim(num_rotations):
            return (
                rotary_dim
                * math.log(orig_max / (num_rotations * 2 * math.pi))
                / (2 * math.log(theta_base))
            )

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001

            ramp = (
                torch.arange(dim, dtype=torch.float32) - min_val
            ) / (max_val - min_val)

            return torch.clamp(ramp, 0, 1)

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), rotary_dim - 1)

        ramp = linear_ramp_factor(low, high, rotary_dim // 2)

        inv_freq = (
            inv_freq_interpolation * ramp
            + inv_freq_extrapolation * (1.0 - ramp)
        )

    elif rope_cfg["rope_type"] != "default":
        raise ValueError(f"Unsupported rope_type: {rope_cfg['rope_type']}")

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles) * attention_factor, torch.sin(angles) * attention_factor


def apply_rope(x, cos, sin, start_pos=0):
    # Partial RoPE: only the first rotary_dim dims of each head are rotated,
    # the remaining dims pass through unchanged.
    batch_size, num_heads, seq_len, head_dim = x.shape
    rotary_dim = cos.shape[-1]

    cos = cos[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)

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

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        x_float = x.float()

        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        return self.weight * (x_float / rms).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.attention_type = cfg["layer_types"][layer_idx]

        if self.attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")

        # The number of query heads differs between layers
        self.num_heads = cfg["num_attention_heads_per_layer"][layer_idx]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("the number of attention heads must be divisible by num_key_value_heads")

        self.sliding_window = cfg["sliding_window"] if self.attention_type == "sliding_attention" else None

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

        # One softplus gate per head scales the attention output before out_proj
        if cfg["gating"] not in (True, "per-head"):
            raise ValueError("only per-head gating is supported")

        self.g_proj = nn.Linear(d_in, self.num_heads, bias=False)

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

        if self.attention_type == "sliding_attention":
            q_pos = torch.arange(past_len, past_len + num_tokens, device=x.device)[:, None]
            k_pos = torch.arange(past_len + num_tokens, device=x.device)[None, :]

            attn_mask = (
                (k_pos <= q_pos)
                & (k_pos > q_pos - self.sliding_window)
            )
            is_causal = False

        elif past_len > 0:
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

        # [B, H, S, D] -> [B, S, H, D]
        context = context.transpose(1, 2)

        gate = F.softplus(self.g_proj(x).float()).to(context.dtype)
        context = context * gate.unsqueeze(-1)

        context = context.contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context), new_cache


class FeedForward(nn.Module):
    # SwiGLU. Used for the dense layer, the shared expert and every routed expert.
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
    # Sigmoid-routed mixture of experts plus a shared expert that every token uses

    def __init__(self, cfg):
        super().__init__()

        if cfg.get("moe_apply_router_weight_on_input", False):
            raise ValueError("moe_apply_router_weight_on_input is not supported")

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.norm_topk_prob = cfg.get("norm_topk_prob", True)
        self.router_logit_softcapping = cfg.get("moe_router_logit_softcapping", 0.0)
        self.routed_scaling_factor = cfg["moe_routed_scaling_factor"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        # Bias used only to pick the experts, never to weight them
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["moe_intermediate_size"])
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert = FeedForward(self.emb_dim, cfg["shared_expert_intermediate_size"])

    def route(self, x_flat):
        logits = self.gate(x_flat).float()

        # Optional tanh soft-capping of the router logits
        if self.router_logit_softcapping > 0.0:
            logits = torch.tanh(logits / self.router_logit_softcapping) * self.router_logit_softcapping

        # Sigmoid scores (not softmax) over the routed experts
        scores = torch.sigmoid(logits)

        topk_indices = torch.topk(scores + self.e_score_correction_bias, self.num_experts_per_tok, dim=-1)[1]

        # Mixing weights come from the original scores, without the bias
        topk_weights = scores.gather(1, topk_indices)

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

        # Only the routed experts are scaled; the shared expert is added as is
        output = routed_out * self.routed_scaling_factor + self.shared_expert(x_flat)

        return output.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = GroupedQueryAttention(cfg, layer_idx)

        if cfg["mlp_layer_types"][layer_idx] == "sparse":
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

        if cfg["num_hidden_layers"] != len(cfg["layer_types"]):
            raise ValueError("layer_types length must equal num_hidden_layers")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # Full-attention and sliding-window layers use different RoPE settings.
        # One table per layer type is shared by every layer of that type.
        self.layer_types = sorted(set(cfg["layer_types"]))

        for layer_type in self.layer_types:
            cos, sin = compute_rope_params(
                cfg["head_dim"],
                cfg["rope_parameters"][layer_type],
                cfg["max_position_embeddings"],
            )
            self.register_buffer(f"cos_{layer_type}", cos, persistent=False)
            self.register_buffer(f"sin_{layer_type}", sin, persistent=False)

        self.layer_rope = list(cfg["layer_types"])

    def forward(self, idx, cache=None, use_cache=False):
        # cache: list with one (K, V) pair per layer, from the previous forward call
        # use_cache=False returns logits only; use_cache=True returns (logits, cache)
        past_len = 0 if cache is None else cache[0][0].size(2)

        if past_len + idx.size(1) > self.cos_full_attention.size(0):
            raise ValueError(
                f"Sequence length {past_len + idx.size(1)} exceeds "
                f"max_position_embeddings {self.cos_full_attention.size(0)}"
            )

        x = self.tok_emb(idx)

        new_cache = []

        for layer_idx, (block, layer_type) in enumerate(zip(self.blocks, self.layer_rope)):
            layer_cache = None if cache is None else cache[layer_idx]
            x, layer_cache = block(
                x, getattr(self, f"cos_{layer_type}"), getattr(self, f"sin_{layer_type}"), layer_cache
            )
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
