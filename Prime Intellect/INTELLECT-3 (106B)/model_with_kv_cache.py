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

        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return x * self.weight


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
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=False)

        if cfg["use_qk_norm"]:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
            self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        else:
            self.q_norm = None
            self.k_norm = None

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

        if self.q_norm is not None:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        Q = apply_rope(Q, cos, sin, start_pos=past_len)
        K = apply_rope(K, cos, sin, start_pos=past_len)

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
    # SwiGLU. Used for the dense layers and as every routed / shared expert.
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

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["n_routed_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_shared_experts = cfg["n_shared_experts"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and n_routed_experts")

        if self.num_experts % self.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")

        # Router only routes among the routed experts
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

        # The shared experts form a single FFN that is n_shared_experts times wider
        self.shared_experts = FeedForward(
            self.emb_dim,
            cfg["moe_intermediate_size"] * self.num_shared_experts,
        )

    def route(self, x_flat):
        # Sigmoid scores (not softmax) over the routed experts
        scores = torch.sigmoid(self.gate(x_flat).float())

        choice_scores = scores + self.e_score_correction_bias

        # Group-limited routing: keep only the best topk_group groups
        # (with n_group = 1 this keeps every expert)
        if self.n_group > 1:
            num_tokens = scores.shape[0]
            group_scores = (
                choice_scores
                .view(num_tokens, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = (
                group_mask
                .unsqueeze(-1)
                .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
                .reshape(num_tokens, self.num_experts)
            )
            choice_scores = choice_scores.masked_fill(score_mask == 0, float("-inf"))

        topk_indices = torch.topk(choice_scores, self.num_experts_per_tok, dim=-1)[1]

        # Mixing weights come from the original scores, without the bias
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_indices

    def forward(self, x):
        batch, seq_len, _ = x.shape

        x_flat = x.reshape(batch * seq_len, self.emb_dim)

        topk_weights, topk_indices = self.route(x_flat)
        topk_weights = topk_weights.to(x_flat.dtype)

        routed_out = torch.zeros_like(x_flat)

        for expert_id in torch.unique(topk_indices).tolist():
            token_idx, slot_idx = torch.where(topk_indices == expert_id)

            expert_out = self.experts[expert_id](x_flat[token_idx])
            expert_out = expert_out * topk_weights[token_idx, slot_idx].unsqueeze(-1)

            routed_out.index_add_(0, token_idx, expert_out)

        # Shared expert is active for every token
        output = routed_out + self.shared_experts(x_flat)

        return output.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = GroupedQueryAttention(cfg)

        # The first first_k_dense_replace layers are dense, the rest are MoE
        if layer_idx < cfg["first_k_dense_replace"]:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])
        else:
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

        if cfg["rope_scaling"] is not None:
            raise ValueError("rope_scaling is not supported")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer
        rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
        cos, sin = compute_rope_params(
            rotary_dim, cfg["rope_theta"], cfg["max_position_embeddings"]
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
                f"the RoPE table length {self.cos.size(0)}"
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
