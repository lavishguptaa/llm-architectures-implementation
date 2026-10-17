import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(rotary_dim, theta_base, context_length, rope_cfg=None):

    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    )

    if rope_cfg is not None:
        if rope_cfg["rope_type"] != "llama3":
            raise ValueError(f"Unsupported rope_type: {rope_cfg['rope_type']}")

        factor = rope_cfg["factor"]
        low_freq_factor = rope_cfg["low_freq_factor"]
        high_freq_factor = rope_cfg["high_freq_factor"]
        orig_max = rope_cfg["original_max_position_embeddings"]

        low_freq_wavelen = orig_max / low_freq_factor
        high_freq_wavelen = orig_max / high_freq_factor

        wavelen = 2 * torch.pi / inv_freq

        smooth = (orig_max / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth) * inv_freq / factor + smooth * inv_freq

        scaled_inv_freq = torch.where(
            wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
        )
        is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, scaled_inv_freq)

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

        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        return ((x_float / rms) * (1.0 + self.weight.float())).to(x.dtype)


class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim, limit=None):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        self.limit = limit

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.attention_type = cfg["layer_types"][layer_idx]

        if self.attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")

        # Sliding-window layers use a different number of query heads
        if self.attention_type == "sliding_attention":
            self.num_heads = cfg["attention_other_setting"]["num_attention_heads"]
            self.num_key_value_heads = cfg["attention_other_setting"]["num_attention_groups"]
            self.sliding_window = cfg["sliding_window"]
        else:
            self.num_heads = cfg["num_attention_heads"]
            self.num_key_value_heads = cfg["num_attention_groups"]
            self.sliding_window = None

        self.head_dim = cfg["head_dim"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_attention_groups")

        eps = cfg.get("rms_norm_eps", 1e-5)

        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=False)

        # Per-head RMSNorm on queries and keys, applied before RoPE
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

        # One sigmoid gate per head scales the attention output before out_proj
        self.use_head_wise_attn_gate = cfg["use_head_wise_attn_gate"]
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(d_in, self.num_heads, bias=False)

    def forward(self, x, cos, sin):
        batch_size, num_tokens, _ = x.shape

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

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        attn_mask = None
        is_causal = True

        if self.attention_type == "sliding_attention":
            positions = torch.arange(num_tokens, device=x.device)
            q_pos = positions[:, None]
            k_pos = positions[None, :]

            attn_mask = (
                (k_pos <= q_pos)
                & (k_pos > q_pos - self.sliding_window)
            )
            is_causal = False

        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True
        )

        # [B, H, S, D] -> [B, S, H, D]
        context = context.transpose(1, 2)

        if self.use_head_wise_attn_gate:
            gate = torch.sigmoid(self.g_proj(x))
            context = context * gate.unsqueeze(-1)

        context = context.contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)


class MoEFeedForward(nn.Module):

    def __init__(self, cfg, limit=None):
        super().__init__()

        if not cfg["use_moe_router_bias"] or cfg["moe_router_activation"] != "sigmoid":
            raise ValueError("only the sigmoid router with a selection bias is supported")

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["moe_num_experts"]
        self.num_experts_per_tok = cfg["moe_top_k"]
        self.routed_scaling_factor = cfg["moe_router_scaling_factor"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("moe_top_k must be between 1 and moe_num_experts")

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        # Bias used only to pick the experts, never to weight them
        self.register_buffer(
            "router_bias",
            torch.zeros(self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["moe_intermediate_size"], limit)
                for _ in range(self.num_experts)
            ]
        )

    def route(self, x_flat):
        # The router runs in float32
        logits = F.linear(x_flat.float(), self.gate.weight.float())

        scores = torch.sigmoid(logits)

        topk_indices = torch.topk(scores + self.router_bias, self.num_experts_per_tok, dim=-1)[1]

        # Mixing weights come from the original scores, without the bias
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

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

        eps = cfg.get("rms_norm_eps", 1e-5)

        self.attn = GroupedQueryAttention(cfg, layer_idx)

        # Clamp limits are given per layer; 0 means no clamp
        limit = cfg["swiglu_limits"][layer_idx] or None
        limit_shared = cfg["swiglu_limits_shared"][layer_idx] or None

        moe_layers = {int(i) for i in cfg["moe_layers_enum"].split(",")}

        self.is_moe = layer_idx in moe_layers

        if self.is_moe:
            self.moe = MoEFeedForward(cfg, limit)
            self.shared_expert = FeedForward(cfg["hidden_size"], cfg["share_expert_dim"], limit_shared)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"], limit_shared)

        self.layer1 = RMSNorm(cfg["hidden_size"], eps=eps)
        self.layer2 = RMSNorm(cfg["hidden_size"], eps=eps)

    def forward(self, x, cos, sin):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.layer2(x)

        if self.is_moe:
            x = self.moe(x) + self.shared_expert(x)
        else:
            x = self.ffn(x)

        x = x + shortcut

        return x


class Model(nn.Module):
    # The multi-token-prediction layers (the config lists 3 extra layers after the 45
    # decoder layers) are not implemented; they are only used for speculative decoding.

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg.get("rms_norm_eps", 1e-5))
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # Every layer has its own RoPE base and rotary fraction, and only the full-attention
        # layers use the llama3 frequency scaling. Layers with the same settings share one
        # table, so only a couple of tables are built.
        yarn_only_types = cfg.get("yarn_only_types")
        table_keys = []
        self.layer_table = []

        for layer_idx in range(cfg["num_hidden_layers"]):
            scaled = yarn_only_types is None or cfg["layer_types"][layer_idx] in yarn_only_types
            key = (
                cfg["rope_theta"][layer_idx],
                cfg["partial_rotary_factors"][layer_idx],
                scaled,
            )

            if key not in table_keys:
                table_keys.append(key)

            self.layer_table.append(table_keys.index(key))

        for table_idx, (theta, partial_factor, scaled) in enumerate(table_keys):
            cos, sin = compute_rope_params(
                int(cfg["head_dim"] * partial_factor),
                theta,
                cfg["max_position_embeddings"],
                cfg["rope_scaling"] if scaled else None,
            )
            self.register_buffer(f"cos_{table_idx}", cos, persistent=False)
            self.register_buffer(f"sin_{table_idx}", sin, persistent=False)

    def forward(self, idx):
        if idx.size(1) > self.cos_0.size(0):
            raise ValueError(
                f"Sequence length {idx.size(1)} exceeds "
                f"max_position_embeddings {self.cos_0.size(0)}"
            )

        x = self.tok_emb(idx)

        for block, table_idx in zip(self.blocks, self.layer_table):
            x = block(x, getattr(self, f"cos_{table_idx}"), getattr(self, f"sin_{table_idx}"))

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
