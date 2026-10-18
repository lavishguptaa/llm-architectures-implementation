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

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: [batch, heads, seq_len, head_dim]; cos / sin: [context_length, head_dim / 2]
    # Adjacent dimension pairs (2i, 2i + 1) are rotated together

    seq_len = x.size(-2)

    cos = cos[:seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[:seq_len, :].to(x.device, dtype=x.dtype)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)


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

        return (x_float / rms).to(x.dtype) * self.weight


def l2_norm(x, eps):
    # Weight-free RMS normalisation, used for QK-norm
    x_float = x.float()

    return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]
        self.scale = self.head_dim ** -0.5

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        # Every no_rope_layer_interval-th layer has no positional encoding (NoPE) and
        # attends to the whole sequence. All other layers use RoPE and chunked attention.
        self.use_rope = (layer_idx + 1) % cfg["no_rope_layer_interval"] != 0

        self.chunk_size = cfg["attention_chunk_size"]
        self.attn_temperature_tuning = cfg["attn_temperature_tuning"]
        self.floor_scale = cfg["floor_scale"]
        self.attn_scale = cfg["attn_scale"]

        self.use_qk_norm = cfg["use_qk_norm"] and self.use_rope
        self.norm_eps = cfg["rms_norm_eps"]

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

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

        if self.use_rope:
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        if self.use_qk_norm:
            Q = l2_norm(Q, self.norm_eps)
            K = l2_norm(K, self.norm_eps)

        # NoPE layers scale the queries up with the position, which keeps attention
        # sharp over very long contexts
        if self.attn_temperature_tuning and not self.use_rope:
            positions = torch.arange(num_tokens, device=x.device).float()

            temperature = (
                torch.log1p(torch.floor((positions + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
            )

            Q = (Q * temperature.view(1, 1, num_tokens, 1)).to(Q.dtype)

        attn_mask = None
        is_causal = True

        # Layers with RoPE only attend inside their own chunk of chunk_size tokens
        if self.use_rope and num_tokens > self.chunk_size:
            positions = torch.arange(num_tokens, device=x.device)
            q_pos = positions[:, None]
            k_pos = positions[None, :]

            attn_mask = (
                (k_pos <= q_pos)
                & (q_pos // self.chunk_size == k_pos // self.chunk_size)
            )
            is_causal = False

        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=True
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)


class FeedForward(nn.Module):
    # SwiGLU. Used for the dense layers, the shared expert and every routed expert.
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
    # Routed experts plus a shared expert that every token goes through

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_local_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")

        self.router = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["intermediate_size"])
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert = FeedForward(self.emb_dim, cfg["intermediate_size"])

    def route(self, x_flat):
        logits = self.router(x_flat)

        # Pick the top-k experts, then turn their logits into sigmoid gates
        topk_logits, topk_indices = torch.topk(logits, self.num_experts_per_tok, dim=-1)
        topk_scores = torch.sigmoid(topk_logits.float())

        return topk_scores, topk_indices

    def forward(self, x):
        batch, seq_len, _ = x.shape

        x_flat = x.reshape(batch * seq_len, self.emb_dim)

        topk_scores, topk_indices = self.route(x_flat)
        topk_scores = topk_scores.to(x_flat.dtype)

        routed_out = torch.zeros_like(x_flat)

        # Run only the experts that received tokens
        for expert_id in torch.unique(topk_indices).tolist():
            token_idx, slot_idx = torch.where(topk_indices == expert_id)

            # The gate scales the expert's input, not its output
            expert_in = x_flat[token_idx] * topk_scores[token_idx, slot_idx].unsqueeze(-1)

            routed_out.index_add_(0, token_idx, self.experts[expert_id](expert_in))

        # Shared expert is active for every token
        output = routed_out + self.shared_expert(x_flat)

        return output.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = GroupedQueryAttention(cfg, layer_idx)

        # MoE and dense layers alternate: every interleave_moe_layer_step-th layer is MoE
        step = cfg["interleave_moe_layer_step"]

        if (layer_idx + 1) % step == 0:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size_mlp"])

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
    # Text model only; the vision encoder of the multimodal checkpoint is not implemented.

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

        # One RoPE table shared by every layer that uses RoPE
        cos, sin = compute_rope_params(
            cfg["head_dim"], cfg["rope_theta"], cfg["max_position_embeddings"]
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

    # The checkpoint config is multimodal: the language model's settings sit under "text_config"
    with open("config.json") as f:
        cfg = json.load(f)["text_config"]

    # These settings are not listed in config.json; they are the defaults of the
    # Hugging Face Llama4TextConfig, which the model was trained with
    cfg.setdefault("no_rope_layer_interval", 4)
    cfg.setdefault("attn_temperature_tuning", True)
    cfg.setdefault("floor_scale", 8192)
    cfg.setdefault("attn_scale", 0.1)
    cfg.setdefault("tie_word_embeddings", False)

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
