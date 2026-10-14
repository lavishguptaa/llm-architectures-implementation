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
    # Gemma scales by (1 + weight), so the weight is zero-initialised
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x_float = x.float()

        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        x_float = x_float / rms

        return (x_float * (1.0 + self.weight.float())).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, attention_type):
        super().__init__()

        if attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        d_in = cfg["hidden_size"]
        d_out = self.num_heads * self.head_dim

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.attention_type = attention_type
        self.sliding_window = cfg["sliding_window"]
        self.attention_dropout_p = cfg["attention_dropout"]
        self.attn_logit_softcapping = cfg["attn_logit_softcapping"]
        self.scale = cfg["query_pre_attn_scalar"] ** -0.5

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_out, d_in, bias=bias)

        cos, sin = compute_rope_params(
            self.head_dim, cfg["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
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

        Q = apply_rope(Q, self.cos, self.sin)
        K = apply_rope(K, self.cos, self.sin)

        # Share each key-value head across its group of query heads
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # Attention is written out because the logits are soft-capped,
        # which scaled_dot_product_attention cannot do
        scores = (Q @ K.transpose(2, 3)) * self.scale

        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping

        positions = torch.arange(num_tokens, device=x.device)
        q_pos = positions[:, None]
        k_pos = positions[None, :]

        mask = k_pos <= q_pos

        if self.attention_type == "sliding_attention":
            mask = mask & (k_pos > q_pos - self.sliding_window)

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
        weights = F.dropout(
            weights, p=self.attention_dropout_p, training=self.training
        )

        context = (weights @ V).transpose(1, 2).contiguous().view(
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
    def __init__(self, cfg, attention_type):
        super().__init__()

        emb_dim = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]

        self.attn = GroupedQueryAttention(cfg, attention_type)
        self.ffn = FeedForward(emb_dim, cfg["intermediate_size"])

        self.input_norm = RMSNorm(emb_dim, eps)
        self.post_attn_norm = RMSNorm(emb_dim, eps)
        self.pre_ffn_norm = RMSNorm(emb_dim, eps)
        self.post_ffn_norm = RMSNorm(emb_dim, eps)

    def forward(self, x):
        shortcut = x
        x = self.input_norm(x)
        x = self.attn(x)
        x = self.post_attn_norm(x)
        x = x + shortcut

        shortcut = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = self.post_ffn_norm(x)
        x = x + shortcut

        return x


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.emb_scale = cfg["hidden_size"] ** 0.5
        self.final_logit_softcapping = cfg["final_logit_softcapping"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        # Layers alternate: sliding_attention, full_attention, sliding_attention, ...
        layer_types = [
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(cfg["num_hidden_layers"])
        ]

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg, attention_type) for attention_type in layer_types]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.out_proj.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx) * self.emb_scale

        x = self.blocks(x)
        x = self.out_norm_layer(x)
        logits = self.out_proj(x)

        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

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
