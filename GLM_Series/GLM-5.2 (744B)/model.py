import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(rope_dim, theta_base, context_length):

    assert rope_dim % 2 == 0, "rope_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, rope_dim, 2).float() / rope_dim)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)

    return torch.cos(angles), torch.sin(angles)


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


def get_indexer_types(cfg):
    # Per layer: "full" runs its own indexer, "shared" reuses the selection of the previous
    # full layer. An explicit list wins, then a pattern string such as "FSSF", then the
    # index_topk_freq / index_skip_topk_offset schedule.
    if cfg.get("indexer_types") is not None:
        return list(cfg["indexer_types"])

    pattern = cfg.get("index_topk_pattern")

    if pattern is not None:
        return [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)

    freq = max(cfg.get("index_topk_freq", 1), 1)
    offset = cfg.get("index_skip_topk_offset", 2)

    return [
        "full" if max(i - offset + 1, 0) % freq == 0 else "shared"
        for i in range(cfg["num_hidden_layers"])
    ]


class Indexer(nn.Module):
    # DeepSeek Sparse Attention's "lightning indexer": a small separate attention that
    # scores every earlier token for each query and keeps only the top index_topk of them.

    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg["index_n_heads"]
        self.head_dim = cfg["index_head_dim"]
        self.rope_dim = cfg["qk_rope_head_dim"]
        self.topk = cfg["index_topk"]

        # Queries come from the main attention's compressed query, keys from the input
        self.wq_b = nn.Linear(cfg["q_lora_rank"], self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg["hidden_size"], self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        # One mixing weight per indexer head, computed from the input
        self.weights_proj = nn.Linear(cfg["hidden_size"], self.n_heads, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, q_latent, cos, sin):
        batch_size, num_tokens, _ = x.shape

        q = self.wq_b(q_latent).view(
            batch_size, num_tokens, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_norm(self.wk(x)).unsqueeze(1)

        # Only the first rope_dim dimensions of an indexer head carry position
        q_rot, q_pass = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_rot, k_pass = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        q = torch.cat([apply_rope(q_rot, cos, sin), q_pass], dim=-1)
        k = torch.cat([apply_rope(k_rot, cos, sin), k_pass], dim=-1)

        # Per-head relevance scores, then a weighted sum over the heads: [B, S, T]
        scores = F.relu(torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale)

        head_weights = self.weights_proj(x).float() * (self.n_heads ** -0.5)
        head_weights = head_weights.transpose(1, 2).unsqueeze(-1)

        index_scores = (scores * head_weights).sum(dim=1)

        positions = torch.arange(num_tokens, device=x.device)
        causal = positions[None, :] <= positions[:, None]

        index_scores = index_scores.masked_fill(~causal, float("-inf"))

        topk = min(self.topk, num_tokens)

        return index_scores.topk(topk, dim=-1).indices


class MultiLatentAttention(nn.Module):
    # Multi-head latent attention whose keys are restricted by the DSA indexer.
    # With indexer sharing, only the "full" layers own an indexer.

    def __init__(self, cfg, indexer_type):
        super().__init__()

        d_model = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg["v_head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]
        self.scale = self.qk_head_dim ** -0.5

        bias = cfg["attention_bias"]

        # The two latent norms use a fixed eps of 1e-6, unlike the block norms
        self.q_down_proj = nn.Linear(d_model, self.q_lora_rank, bias=bias)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=1e-6)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, d_model, bias=bias)

        # A "full" layer runs its own indexer; a "shared" layer reuses the selection of the
        # previous full layer, so it has no indexer at all
        self.indexer = Indexer(cfg) if indexer_type == "full" else None

    def forward(self, x, cos, sin, prev_topk_indices=None):
        batch_size, seq_len, _ = x.shape

        q_latent = self.q_norm(self.q_down_proj(x))

        q = self.q_up_proj(q_latent)
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

        # DSA: each query may only attend to the tokens selected by an indexer
        if self.indexer is not None:
            topk_indices = self.indexer(x, q_latent, cos, sin)
        else:
            topk_indices = prev_topk_indices

        selected = torch.zeros(
            batch_size, seq_len, seq_len, dtype=torch.bool, device=x.device
        ).scatter(-1, topk_indices, True)

        positions = torch.arange(seq_len, device=x.device)
        causal = positions[None, :] <= positions[:, None]

        attn_mask = (selected & causal).unsqueeze(1)

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.v_head_dim
        )

        return self.out_proj(context), topk_indices


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

        if cfg["scoring_func"] != "sigmoid" or cfg["topk_method"] != "noaux_tc":
            raise ValueError("only sigmoid scoring with topk_method 'noaux_tc' is supported")

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
        # The router runs in float32; sigmoid scores (not softmax) over the routed experts
        scores = torch.sigmoid(F.linear(x_flat.float(), self.gate.weight.float()))

        choice_scores = scores + self.e_score_correction_bias

        # Group-limited routing: a group is scored by its two best experts and only
        # the best topk_group groups stay in the running
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

        self.attn = MultiLatentAttention(cfg, get_indexer_types(cfg)[layer_idx])

        # mlp_layer_types lists which layers are dense; without it the first
        # first_k_dense_replace layers are dense and the rest are MoE
        mlp_types = cfg.get("mlp_layer_types")
        is_dense = (
            mlp_types[layer_idx] == "dense" if mlp_types is not None
            else layer_idx < cfg["first_k_dense_replace"]
        )

        if is_dense:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])
        else:
            self.ffn = MoEFeedForward(cfg)

        self.layer1 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.layer2 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, prev_topk_indices=None):
        shortcut = x
        x = self.layer1(x)
        x, topk_indices = self.attn(x, cos, sin, prev_topk_indices)
        x = x + shortcut

        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, topk_indices


class Model(nn.Module):
    # The multi-token-prediction module (num_nextn_predict_layers) is not implemented;
    # it is only used for speculative decoding and training.

    def __init__(self, cfg):
        super().__init__()

        if cfg["rope_parameters"]["rope_type"] != "default":
            raise ValueError("only the default RoPE is supported")

        # A shared layer needs a selection from an earlier layer, so layer 0 must run its own indexer
        if get_indexer_types(cfg)[0] != "full":
            raise ValueError("the first layer must be a full indexer layer")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer (and by the indexers)
        cos, sin = compute_rope_params(
            cfg["qk_rope_head_dim"],
            cfg["rope_parameters"]["rope_theta"],
            cfg["max_position_embeddings"],
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

        topk_indices = None

        for block in self.blocks:
            x, topk_indices = block(x, self.cos, self.sin, topk_indices)

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
