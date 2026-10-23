import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(head_dim, rope_cfg, context_length):

    theta_base = rope_cfg["rope_theta"]

    if rope_cfg["rope_type"] == "default":
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

    elif rope_cfg["rope_type"] == "proportional":

        rope_angles = int(rope_cfg["partial_rotary_factor"] * head_dim // 2)

        inv_freq_rotated = 1.0 / (
            theta_base ** (torch.arange(0, 2 * rope_angles, 2).float() / head_dim)
        )

        inv_freq = torch.cat(
            [inv_freq_rotated, torch.zeros(head_dim // 2 - rope_angles)]
        )

    else:
        raise ValueError(f"Unsupported rope_type: {rope_cfg['rope_type']}")

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):

    batch_size, num_heads, seq_len, head_dim = x.shape

    cos = cos[:seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[:seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


class RMSNorm(nn.Module):

    def __init__(self, emb_dim, eps, with_scale=True):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(emb_dim)) if with_scale else None

    def forward(self, x):
        x_float = x.float()

        x_norm = x_float * torch.pow(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps, -0.5
        )

        if self.weight is not None:
            x_norm = x_norm * self.weight.float()

        return x_norm.to(x.dtype)


def kv_sharing_plan(layer_types, num_kv_shared_layers):
    
    num_layers = len(layer_types)
    first_shared = num_layers - num_kv_shared_layers

    plan = []

    for layer_idx, layer_type in enumerate(layer_types):
        is_shared = layer_idx >= first_shared > 0

        if first_shared > 0:
            earlier = layer_types[:first_shared]
            last_of_type = len(earlier) - 1 - earlier[::-1].index(layer_type)
        else:
            last_of_type = None

        if is_shared:
            plan.append((last_of_type, False))
        else:
            plan.append((None, layer_idx == last_of_type))

    return plan


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx, attention_type, shared_source, stores_kv):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.layer_idx = layer_idx

        if attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.attention_type = attention_type
        self.is_sliding = attention_type == "sliding_attention"
        self.sliding_window = cfg["sliding_window"] if self.is_sliding else None

        self.num_heads = cfg["num_attention_heads"]
        self.attention_dropout_p = cfg["attention_dropout"]

        # Full-attention layers use a wider head
        self.head_dim = cfg["head_dim"] if self.is_sliding else cfg["global_head_dim"]

        # With attention_k_eq_v the keys double as the values on full-attention layers
        self.k_eq_v = cfg["attention_k_eq_v"] and not self.is_sliding

        self.num_key_value_heads = (
            cfg["num_global_key_value_heads"] if self.k_eq_v else cfg["num_key_value_heads"]
        )

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by the number of key/value heads")

        # Layers that reuse another layer's keys and values have no K/V weights of their own
        self.shared_source = shared_source
        self.stores_kv = stores_kv

        bias = cfg["attention_bias"]
        eps = cfg["rms_norm_eps"]

        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps)

        if shared_source is None:
            self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
            self.W_v = None if self.k_eq_v else nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
            self.k_norm = RMSNorm(self.head_dim, eps)
            self.v_norm = RMSNorm(self.head_dim, eps, with_scale=False)

    def forward(self, x, cos, sin, shared_kv):
        batch_size, num_tokens, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        Q = apply_rope(self.q_norm(Q).transpose(1, 2), cos, sin)

        if self.shared_source is not None:
            K, V = shared_kv[self.shared_source]
        else:
            K = self.W_k(x).view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)

            # With k_eq_v the values are the raw key projection (before k_norm and RoPE)
            V = K if self.k_eq_v else self.W_v(x).view(
                batch_size, num_tokens, self.num_key_value_heads, self.head_dim
            )

            K = apply_rope(self.k_norm(K).transpose(1, 2), cos, sin)
            V = self.v_norm(V).transpose(1, 2)

        if self.stores_kv:
            shared_kv[self.layer_idx] = (K, V)

        attn_mask = None
        is_causal = True

        if self.sliding_window is not None:
            positions = torch.arange(num_tokens, device=x.device)
            q_pos = positions[:, None]
            k_pos = positions[None, :]

            attn_mask = (
                (k_pos <= q_pos)
                & (k_pos > q_pos - self.sliding_window)
            )
            is_causal = False

        # No 1/sqrt(head_dim) scaling: the query/key norms already fix the scale
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0,
            enable_gqa=True
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)


class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        value = self.up_proj(x)

        return self.down_proj(gate * value)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx, attention_type, shared_source, stores_kv):
        super().__init__()

        emb_dim = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]

        self.attn = GroupedQueryAttention(cfg, layer_idx, attention_type, shared_source, stores_kv)

        # The layers that reuse keys/values can have a twice as wide MLP (use_double_wide_mlp)
        is_kv_shared = shared_source is not None
        width_factor = 2 if (cfg["use_double_wide_mlp"] and is_kv_shared) else 1

        self.ffn = FeedForward(emb_dim, cfg["intermediate_size"] * width_factor)

        self.input_norm = RMSNorm(emb_dim, eps)
        self.post_attn_norm = RMSNorm(emb_dim, eps)
        self.pre_ffn_norm = RMSNorm(emb_dim, eps)
        self.post_ffn_norm = RMSNorm(emb_dim, eps)

        # A learned scalar that scales the whole block output (stored as a buffer; 1 until loaded)
        self.register_buffer("layer_scalar", torch.ones(1))

        # Per-layer input: a small gated side path fed by a token-specific embedding
        self.per_layer_dim = cfg["hidden_size_per_layer_input"]

        if self.per_layer_dim:
            self.per_layer_input_gate = nn.Linear(emb_dim, self.per_layer_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.per_layer_dim, emb_dim, bias=False)
            self.post_per_layer_input_norm = RMSNorm(emb_dim, eps)

    def forward(self, x, cos, sin, per_layer_input, shared_kv):
        shortcut = x
        x = self.input_norm(x)
        x = self.attn(x, cos, sin, shared_kv)
        x = self.post_attn_norm(x)
        x = x + shortcut

        shortcut = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = self.post_ffn_norm(x)
        x = x + shortcut

        if self.per_layer_dim:
            shortcut = x
            x = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
            x = self.per_layer_projection(x)
            x = self.post_per_layer_input_norm(x)
            x = x + shortcut

        return x * self.layer_scalar


class Model(nn.Module):
    # Text decoder only: the vision and audio towers and the multimodal wrapper are not implemented.
    # The mixture-of-experts block of the larger Gemma 4 sizes is not implemented either.

    def __init__(self, cfg):
        super().__init__()

        if cfg["enable_moe_block"]:
            raise ValueError("enable_moe_block is not supported")

        if cfg["num_hidden_layers"] != len(cfg["layer_types"]):
            raise ValueError("layer_types length must equal num_hidden_layers")

        self.final_logit_softcapping = cfg["final_logit_softcapping"]
        self.num_layers = cfg["num_hidden_layers"]
        self.per_layer_dim = cfg["hidden_size_per_layer_input"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        # Gemma multiplies the token embeddings by sqrt(hidden_size)
        self.register_buffer(
            "emb_scale", torch.tensor(cfg["hidden_size"] ** 0.5), persistent=False
        )

        plan = kv_sharing_plan(cfg["layer_types"], cfg["num_kv_shared_layers"])

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg, layer_idx, attention_type, *plan[layer_idx])
                for layer_idx, attention_type in enumerate(cfg["layer_types"])
            ]
        )

        if self.per_layer_dim:
            # Every token also gets a small embedding for every layer, mixed with a projection
            # of its main embedding
            total = self.num_layers * self.per_layer_dim

            self.tok_emb_per_layer = nn.Embedding(cfg["vocab_size_per_layer_input"], total)
            self.register_buffer(
                "per_layer_emb_scale", torch.tensor(self.per_layer_dim ** 0.5), persistent=False
            )
            self.per_layer_model_projection = nn.Linear(cfg["hidden_size"], total, bias=False)
            self.per_layer_projection_norm = RMSNorm(self.per_layer_dim, cfg["rms_norm_eps"])
            self.per_layer_projection_scale = cfg["hidden_size"] ** -0.5
            self.per_layer_input_scale = 2.0 ** -0.5

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

       
        self.layer_types = list(cfg["layer_types"])

        head_dims = {
            "sliding_attention": cfg["head_dim"],
            "full_attention": cfg["global_head_dim"],
        }

        for layer_type in sorted(set(self.layer_types)):
            cos, sin = compute_rope_params(
                head_dims[layer_type],
                cfg["rope_parameters"][layer_type],
                cfg["max_position_embeddings"],
            )
            self.register_buffer(f"cos_{layer_type}", cos, persistent=False)
            self.register_buffer(f"sin_{layer_type}", sin, persistent=False)

    def per_layer_inputs(self, idx, x):
        # [B, S, num_layers, per_layer_dim]
        batch_size, num_tokens = idx.shape

        from_tokens = (
            self.tok_emb_per_layer(idx) * self.per_layer_emb_scale.to(self.tok_emb.weight.dtype)
        ).view(batch_size, num_tokens, self.num_layers, self.per_layer_dim)

        from_embeddings = (
            self.per_layer_model_projection(x) * self.per_layer_projection_scale
        ).view(batch_size, num_tokens, self.num_layers, self.per_layer_dim)

        from_embeddings = self.per_layer_projection_norm(from_embeddings)

        return (from_embeddings + from_tokens) * self.per_layer_input_scale

    def forward(self, idx):
        if idx.size(1) > self.cos_full_attention.size(0):
            raise ValueError(
                f"Sequence length {idx.size(1)} exceeds "
                f"max_position_embeddings {self.cos_full_attention.size(0)}"
            )

        x = self.tok_emb(idx) * self.emb_scale.to(self.tok_emb.weight.dtype)

        per_layer = self.per_layer_inputs(idx, x) if self.per_layer_dim else None

        # Keys and values saved by the layers that other layers will reuse
        shared_kv = {}

        for layer_idx, (block, layer_type) in enumerate(zip(self.blocks, self.layer_types)):
            x = block(
                x,
                getattr(self, f"cos_{layer_type}"),
                getattr(self, f"sin_{layer_type}"),
                per_layer[:, :, layer_idx, :] if per_layer is not None else None,
                shared_kv,
            )

        x = self.out_norm_layer(x)
        logits = self.out_proj(x)

        # Soft-cap the final logits to (-cap, cap)
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap

        return logits


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
