import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def yarn_get_mscale(scale, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_rope_params(rope_dim, cfg):
    # YaRN-scaled RoPE for the rotary part of each head (rope_dim = qk_rope_head_dim)

    theta_base = cfg["rope_theta"]
    context_length = cfg["max_position_embeddings"]

    rope_cfg = cfg["rope_scaling"]
    factor = rope_cfg["factor"]
    orig_max = rope_cfg["original_max_position_embeddings"]
    beta_fast = rope_cfg["beta_fast"]
    beta_slow = rope_cfg["beta_slow"]
    mscale = rope_cfg.get("mscale", 1.0)
    mscale_all_dim = rope_cfg.get("mscale_all_dim", 0.0)

    assert rope_dim % 2 == 0, "rope_dim must be even"

    # Scale applied to cos / sin
    attention_factor = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)

    def find_correction_dim(num_rotations):
        return (
            rope_dim
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

    pos_freqs = theta_base ** (torch.arange(0, rope_dim, 2).float() / rope_dim)

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), rope_dim - 1)

    # ramp is 0 for high-frequency dims (kept as is) and 1 for low-frequency dims (interpolated)
    ramp = linear_ramp_factor(low, high, rope_dim // 2)

    inv_freq = (
        inv_freq_interpolation * ramp
        + inv_freq_extrapolation * (1.0 - ramp)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)

    cos = angles.cos() * attention_factor
    sin = angles.sin() * attention_factor

    return cos, sin


def apply_rope(x, cos, sin):
    # x: [batch, heads, seq_len, rope_dim]; cos / sin: [context_length, rope_dim / 2]
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

        return self.weight * (x_float / rms).to(x.dtype)


class MultiLatentAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg["q_lora_rank"] is None:
            raise ValueError("q_lora_rank must be set")

        d_model = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg["v_head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]

        bias = cfg["attention_bias"]

        # Queries: down-project, normalise, up-project to [nope | rope] for every head
        self.q_down_proj = nn.Linear(d_model, self.q_lora_rank, bias=bias)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=cfg["rms_norm_eps"])
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # Keys / values: one shared latent plus a single shared rotary key
        self.kv_down_proj = nn.Linear(d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=cfg["rms_norm_eps"])
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, d_model, bias=bias)

        # YaRN also rescales the attention logits by mscale ** 2
        self.scale = self.qk_head_dim ** -0.5

        rope_cfg = cfg["rope_scaling"]
        if rope_cfg is not None and rope_cfg.get("mscale_all_dim"):
            mscale = yarn_get_mscale(rope_cfg["factor"], rope_cfg["mscale_all_dim"])
            self.scale = self.scale * mscale * mscale

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape

        q = self.q_up_proj(self.q_norm(self.q_down_proj(x)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_down_proj(x)
        latent, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        kv = self.kv_up_proj(self.kv_norm(latent))
        kv = kv.view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Shared rotary key: rotate the single head once, then broadcast across heads
        k_rope = k_rope.unsqueeze(1)

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=True,
            scale=self.scale
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.v_head_dim
        )

        return self.out_proj(context)


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

        if cfg["scoring_func"] != "softmax" or cfg["topk_method"] != "group_limited_greedy":
            raise ValueError("only softmax scoring with topk_method 'group_limited_greedy' is supported")

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
        # Softmax scores over the routed experts
        scores = torch.softmax(self.gate(x_flat).float(), dim=-1)

        # Group-limited routing: a group is scored by its best expert and only
        # the best topk_group groups stay in the running
        num_tokens = scores.shape[0]
        group_scores = scores.view(num_tokens, self.n_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, self.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask
            .unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, self.num_experts)
        )
        masked_scores = scores.masked_fill(score_mask == 0, 0.0)

        topk_weights, topk_indices = torch.topk(masked_scores, self.num_experts_per_tok, dim=-1)

        # As in DeepSeek's reference code, renormalising the weights replaces
        # (rather than adds to) the routed scaling factor
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        else:
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

        # Shared expert is active for every token
        output = routed_out + self.shared_experts(x_flat)

        return output.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = MultiLatentAttention(cfg)

        # The first first_k_dense_replace layers are dense; after that every
        # moe_layer_freq-th layer is a mixture of experts
        is_moe = (
            layer_idx >= cfg["first_k_dense_replace"]
            and layer_idx % cfg["moe_layer_freq"] == 0
        )

        if is_moe:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])

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
    # The multi-token-prediction module (num_nextn_predict_layers) is not implemented;
    # it is only used for speculative decoding and training.

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer
        cos, sin = compute_rope_params(cfg["qk_rope_head_dim"], cfg)
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
