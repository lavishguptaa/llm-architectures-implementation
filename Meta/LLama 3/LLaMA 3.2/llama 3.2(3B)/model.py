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

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
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

        # Long wavelengths are slowed down by `factor`, short ones are kept,
        # and the band in between is blended smoothly
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

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=True,
            enable_gqa=True
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg["hidden_size"]
        hidden_dim = cfg["intermediate_size"]
        bias = cfg["mlp_bias"]

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=bias)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        return self.down_proj(gate * value)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = GroupedQueryAttention(cfg)
        self.ffn = FeedForward(cfg)

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

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer
        cos, sin = compute_rope_params(cfg["head_dim"], cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
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

    with open("config.json") as f:
        cfg = json.load(f)

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()

    input_text = "Hello, how are you?"

    tokenizer = Tokenizer.from_file("tokenizer.json")

    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != cfg["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match "
            f"model vocab_size ({cfg['vocab_size']})."
        )

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
