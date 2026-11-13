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

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # The normalisation runs in float32, the result goes back to the input dtype
        input_dtype = x.dtype
        x = x.float()

        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return self.weight * x.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    # MiMo-V2 has two kinds of attention layer that share this code but not their sizes:
    #   full attention  : every token sees all earlier tokens
    #   sliding window  : every token sees only the last `sliding_window` tokens (itself included)
    # Keys / queries are head_dim wide, values are v_head_dim wide (smaller), so the output
    # of every head is v_head_dim wide.
    def __init__(self, cfg, is_swa):
        super().__init__()

        d_in = cfg["hidden_size"]

        if is_swa:
            self.num_heads = cfg["swa_num_attention_heads"]
            self.num_key_value_heads = cfg["swa_num_key_value_heads"]
            self.head_dim = cfg["swa_head_dim"]
            self.v_head_dim = cfg["swa_v_head_dim"]
            self.sliding_window = cfg["sliding_window"]
            use_sink = cfg["add_swa_attention_sink_bias"]
        else:
            self.num_heads = cfg["num_attention_heads"]
            self.num_key_value_heads = cfg["num_key_value_heads"]
            self.head_dim = cfg["head_dim"]
            self.v_head_dim = cfg["v_head_dim"]
            self.sliding_window = None
            use_sink = cfg["add_full_attention_sink_bias"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.attention_dropout_p = cfg["attention_dropout"]
        self.scale = self.head_dim ** -0.5

        # Values are multiplied by a constant before attention (None = no scaling)
        self.value_scale = cfg["attention_value_scale"]

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim

        bias = cfg["attention_bias"]

        # "fused_qkv"
        self.fused_qkv = cfg.get("attention_projection_layout", "split") == "fused_qkv"

        if self.fused_qkv:
            self.qkv_proj = nn.Linear(d_in, self.q_size + self.k_size + self.v_size, bias=bias)
        else:
            self.W_q = nn.Linear(d_in, self.q_size, bias=bias)
            self.W_k = nn.Linear(d_in, self.k_size, bias=bias)
            self.W_v = nn.Linear(d_in, self.v_size, bias=bias)

        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, d_in, bias=False)

        # One learned "sink" logit per head. It joins the softmax like an extra key that has
        # no value, so a head can put weight on nothing. It is a frozen parameter (inference only).
        if use_sink:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)
        else:
            self.sinks = None

    def project(self, x):
        batch_size, num_tokens, _ = x.shape

        if self.fused_qkv:
            Q, K, V = self.qkv_proj(x).split([self.q_size, self.k_size, self.v_size], dim=-1)
        else:
            Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tokens, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

        if self.value_scale is not None:
            V = V * self.value_scale

        return Q, K, V

    def attend(self, Q, K, V, past_len=0):
        # Q: [batch, heads, new_tokens, head_dim]; K, V cover past_len + new_tokens positions
        num_tokens = Q.size(2)
        total = K.size(2)
        dropout_p = self.attention_dropout_p if self.training else 0.0

        needs_mask = self.sliding_window is not None or past_len > 0
        mask = None

        if needs_mask:
            q_pos = torch.arange(past_len, past_len + num_tokens, device=Q.device)[:, None]
            k_pos = torch.arange(total, device=Q.device)[None, :]

            mask = k_pos <= q_pos

            if self.sliding_window is not None:
                mask = mask & (k_pos > q_pos - self.sliding_window)

        if self.sinks is None:
            return F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=mask is None,
                scale=self.scale,
                enable_gqa=True
            )

        # Attention with a sink: written out so the sink column can join the softmax
        groups = self.num_heads // self.num_key_value_heads
        K = K.repeat_interleave(groups, dim=1)
        V = V.repeat_interleave(groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is None:
            positions = torch.arange(num_tokens, device=Q.device)
            mask = positions[None, :] <= positions[:, None]

        scores = scores.masked_fill(~mask, float("-inf"))

        sinks = self.sinks.reshape(1, -1, 1, 1).expand(Q.size(0), -1, num_tokens, 1)
        logits = torch.cat([scores, sinks.to(scores.dtype)], dim=-1)

        # Take the softmax with the sink, then drop its column: it only absorbs probability mass
        weights = torch.softmax(logits.float(), dim=-1).to(V.dtype)[..., :-1]
        weights = F.dropout(weights, p=dropout_p, training=self.training)

        return torch.matmul(weights, V)

    def forward(self, x, cos, sin):
        batch_size, num_tokens, _ = x.shape

        Q, K, V = self.project(x)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        context = self.attend(Q, K, V)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.v_head_dim
        )

        return self.out_proj(context)


class FeedForward(nn.Module):
    
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Router(nn.Module):
    # Sigmoid scores with a selection-only bias ("noaux_tc" routing, as in DeepSeek V3)
    def __init__(self, cfg):
        super().__init__()

        if cfg["scoring_func"] != "sigmoid":
            raise ValueError("only sigmoid scoring is supported")
        if cfg["topk_method"] != "noaux_tc":
            raise ValueError("only noaux_tc routing is supported")

        self.num_experts = cfg["n_routed_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]

        scaling = cfg["routed_scaling_factor"]
        self.routed_scaling_factor = 1.0 if scaling is None else scaling

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and n_routed_experts")

        if self.num_experts % self.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")

        self.weight = nn.Parameter(torch.zeros(self.num_experts, cfg["hidden_size"]))

        # Bias used only to pick the experts, never to weight them (frozen, inference only)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts), requires_grad=False
        )

    def forward(self, x_flat):
        # The router always runs in float32
        scores = torch.sigmoid(F.linear(x_flat.float(), self.weight.float()))

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

        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return topk_weights * self.routed_scaling_factor, topk_indices


class MoEFeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if cfg["n_shared_experts"] is not None:
            raise ValueError("shared experts are not supported (MiMo-V2 has none)")

        self.emb_dim = cfg["hidden_size"]

        self.gate = Router(cfg)

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["moe_intermediate_size"])
                for _ in range(cfg["n_routed_experts"])
            ]
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape

        x_flat = x.reshape(batch * seq_len, self.emb_dim)

        topk_weights, topk_indices = self.gate(x_flat)
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

        # hybrid_layer_pattern: 1 = sliding-window layer, 0 = full-attention layer
        self.is_swa = cfg["hybrid_layer_pattern"][layer_idx] == 1

        self.attn = GroupedQueryAttention(cfg, self.is_swa)

        # moe_layer_freq: 1 = mixture-of-experts layer, 0 = dense layer
        if cfg["n_routed_experts"] is not None and cfg["moe_layer_freq"][layer_idx]:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])

        self.layer1 = RMSNorm(cfg["hidden_size"], eps=cfg["layernorm_epsilon"])
        self.layer2 = RMSNorm(cfg["hidden_size"], eps=cfg["layernorm_epsilon"])

    def forward(self, x, rope_full, rope_swa):
        cos, sin = rope_swa if self.is_swa else rope_full

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

    def __init__(self, cfg):
        super().__init__()

        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is not None and rope_scaling.get("rope_type", rope_scaling.get("type")) != "default":
            raise ValueError("only the default (unscaled) RoPE is supported")

        if len(cfg["hybrid_layer_pattern"]) != cfg["num_hidden_layers"]:
            raise ValueError("hybrid_layer_pattern needs one entry per layer")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["layernorm_epsilon"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # Two RoPE tables: full-attention layers and sliding-window layers use different
        # base frequencies (rope_theta and swa_rope_theta). Both rotate only part of each head.
        full_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
        swa_dim = int(cfg["swa_head_dim"] * cfg["partial_rotary_factor"])

        cos, sin = compute_rope_params(full_dim, cfg["rope_theta"], cfg["max_position_embeddings"])
        self.register_buffer("cos_full", cos, persistent=False)
        self.register_buffer("sin_full", sin, persistent=False)

        cos, sin = compute_rope_params(swa_dim, cfg["swa_rope_theta"], cfg["max_position_embeddings"])
        self.register_buffer("cos_swa", cos, persistent=False)
        self.register_buffer("sin_swa", sin, persistent=False)

    def forward(self, idx):
        x = self.tok_emb(idx)

        rope_full = (self.cos_full, self.sin_full)
        rope_swa = (self.cos_swa, self.sin_swa)

        for block in self.blocks:
            x = block(x, rope_full, rope_swa)

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
