import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(head_dim, cfg):

    theta_base = cfg["rope_theta"]
    context_length = cfg["max_position_embeddings"]

    rope_cfg = cfg["rope_scaling"]

    assert head_dim % 2 == 0, "head_dim must be even"

    if rope_cfg["rope_type"] != "yarn":
        raise ValueError(f"Unsupported rope_type: {rope_cfg['rope_type']}")

    factor = rope_cfg["factor"]
    orig_max = rope_cfg["original_max_position_embeddings"]
    beta_fast = rope_cfg["beta_fast"]
    beta_slow = rope_cfg["beta_slow"]
    truncate = rope_cfg.get("truncate", True)

    # YaRN also scales cos / sin
    attention_factor = 0.1 * math.log(factor) + 1.0

    def find_correction_dim(num_rotations):
        return (
            head_dim
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

    pos_freqs = theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)

    # With truncate = false the (fractional) bounds are used as they are
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)

    low = max(low, 0)
    high = min(high, head_dim - 1)

    # ramp is 0 for high-frequency dims (kept as is) and 1 for low-frequency dims (interpolated)
    ramp = linear_ramp_factor(low, high, head_dim // 2)

    inv_freq = (
        inv_freq_interpolation * ramp
        + inv_freq_extrapolation * (1.0 - ramp)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles) * attention_factor, torch.sin(angles) * attention_factor


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

        return (self.weight * (x_float / rms)).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.attention_type = cfg["layer_types"][layer_idx]

        if self.attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]
        self.scale = self.head_dim ** -0.5

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.sliding_window = cfg["sliding_window"] if self.attention_type == "sliding_attention" else None

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

        self.sinks = nn.Parameter(torch.zeros(self.num_heads))

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

        Q = apply_rope(Q, cos, sin, start_pos=past_len)
        K = apply_rope(K, cos, sin, start_pos=past_len)

        # Keys are cached after RoPE, so they are never recomputed
        if layer_cache is not None:
            K = torch.cat([layer_cache[0], K], dim=2)
            V = torch.cat([layer_cache[1], V], dim=2)

        new_cache = (K, V)

        # Share each key / value head between the query heads of its group
        groups = self.num_heads // self.num_key_value_heads
        K = K.repeat_interleave(groups, dim=1)
        V = V.repeat_interleave(groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # The queries are the new tokens; the keys are the whole sequence, cached keys included
        q_pos = torch.arange(past_len, past_len + num_tokens, device=x.device)[:, None]
        k_pos = torch.arange(past_len + num_tokens, device=x.device)[None, :]

        mask = k_pos <= q_pos

        if self.sliding_window is not None:
            mask = mask & (k_pos > q_pos - self.sliding_window)

        scores = scores.masked_fill(~mask, float("-inf"))

        sinks = self.sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, num_tokens, 1)
        logits = torch.cat([scores, sinks.to(scores.dtype)], dim=-1)

        weights = torch.softmax(logits.float(), dim=-1).to(V.dtype)[..., :-1]
        weights = F.dropout(weights, p=self.attention_dropout_p, training=self.training)

        context = torch.matmul(weights, V)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context), new_cache


class Expert(nn.Module):
    
    def __init__(self, emb_dim, hidden_dim, limit, alpha=1.702):
        super().__init__()

        self.gate_up_proj = nn.Linear(emb_dim, 2 * hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=True)

        self.limit = limit
        self.alpha = alpha

    def forward(self, x):
        gate_up = self.gate_up_proj(x)

        gate = gate_up[..., ::2].clamp(max=self.limit)
        up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)

        # Clamped SwiGLU variant: gate * sigmoid(alpha * gate), times (up + 1)
        glu = gate * torch.sigmoid(gate * self.alpha)

        return self.down_proj((up + 1) * glu)


class MoEFeedForward(nn.Module):
    # Mixture of experts without a shared expert. Every layer is a MoE layer.

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_local_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")

        self.router = nn.Linear(self.emb_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList(
            [
                Expert(self.emb_dim, cfg["intermediate_size"], cfg["swiglu_limit"])
                for _ in range(self.num_experts)
            ]
        )

    def route(self, x_flat):
        logits = self.router(x_flat)

        # Pick the top-k experts on the raw logits, then softmax over just those k
        topk_logits, topk_indices = torch.topk(logits, self.num_experts_per_tok, dim=-1)
        topk_weights = torch.softmax(topk_logits.float(), dim=-1)

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

        self.attn = GroupedQueryAttention(cfg, layer_idx)
        self.ffn = MoEFeedForward(cfg)

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

        # One RoPE table shared by every layer
        cos, sin = compute_rope_params(cfg["head_dim"], cfg)
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
