import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, data, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(data).ids

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    data,
    tokenizer,
    context_size=256,
    stride=128,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    drop_last=False
):
    if tokenizer is None:
        raise ValueError("tokenizer must be provided; load tokenizer.json before creating the dataloader")
    dataset = CustomDataset(data, tokenizer, context_size, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=drop_last
    )
    return dataloader


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        x = x / rms

        return x * self.weight


def compute_rope_params(head_dim, cfg):
    # YaRN-scaled RoPE for the rotary part of each head (head_dim = qk_rope_head_dim)

    theta_base = cfg["rope_theta"]
    context_length = cfg["context_length"]

    rope_cfg = cfg["rope_scaling"]
    factor = rope_cfg["factor"]
    orig_max = rope_cfg["original_max_position_embeddings"]
    beta_fast = rope_cfg["beta_fast"]
    beta_slow = rope_cfg["beta_slow"]
    mscale = rope_cfg["mscale"]
    mscale_all_dim = rope_cfg["mscale_all_dim"]

    assert head_dim % 2 == 0, "head_dim must be even"

    def get_mscale(scale, mscale=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if mscale and mscale_all_dim:
        attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
    else:
        attention_factor = get_mscale(factor)

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

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), head_dim - 1)

    # ramp is 0 for high-frequency dims (kept as is) and 1 for low-frequency dims (interpolated)
    ramp = linear_ramp_factor(low, high, head_dim // 2)

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


class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                 attention_dropout, rms_norm_eps, llama_4_scaling_beta,
                 original_max_position_embeddings):
        super().__init__()
        assert qk_rope_head_dim % 2 == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attention_dropout = attention_dropout
        self.llama_4_scaling_beta = llama_4_scaling_beta
        self.original_max_position_embeddings = original_max_position_embeddings


        self.q_down_proj = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_up_proj = nn.Linear(q_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        
        self.q_rope_proj = nn.Linear(q_lora_rank,n_heads * qk_rope_head_dim,bias=False)


        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            n_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        

        self.k_rope_proj = nn.Linear(d_model,qk_rope_head_dim,bias=False,)

        self.W_o = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape


        c_q = self.q_norm(self.q_down_proj(x))
        q_nope = self.q_up_proj(c_q).view(
            batch_size, seq_len, self.n_heads, self.qk_nope_head_dim
        ).transpose(1,2)
        
        q_rope = self.q_rope_proj(c_q).view(batch_size,seq_len,self.n_heads,self.qk_rope_head_dim).transpose(1,2)
        


        c_kv = self.kv_norm(self.kv_down_proj(x))
        kv = self.kv_up_proj(c_kv).view(
            batch_size, seq_len, self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Shared latent K_rope: rotate the single head once, then broadcast across heads
        k_rope = self.k_rope_proj(x)
        k_rope = k_rope.unsqueeze(1)

        # RoPE only on the rope dimensions
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Long-context query scaling: exactly 1 until original_max_position_embeddings
        positions = torch.arange(seq_len, device=x.device).float()
        q_scale = 1.0 + self.llama_4_scaling_beta * torch.log(
            1.0 + torch.floor(positions / self.original_max_position_embeddings)
        )
        q = q * q_scale.view(1, 1, seq_len, 1).to(q.dtype)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=(
                   self.attention_dropout
                    if self.training
                    else 0.0
            ),
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.v_head_dim
        )
        return self.W_o(output)


class SwiGLUExpert(nn.Module):

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

        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"]

        # Routed experts
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]

        self.num_shared_experts = cfg["num_shared_experts"]

        self.shared_expert_hidden_dim = cfg["shared_expert_hidden_dim"]
        
        self.routed_scaling_factor = cfg['routed_scaling_factor']

        if self.num_experts <= 0:
            raise ValueError("num_experts must be greater than 0.")

        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts cannot be negative.")

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_experts."
            )

        # Router only routes among routed experts.
        self.gate = nn.Linear(
            self.emb_dim,
            self.num_experts,
            bias=False,
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    self.emb_dim,
                    self.hidden_dim,
                )
                for _ in range(self.num_experts)
            ]
        )


        self.shared_experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    self.emb_dim,
                    self.shared_expert_hidden_dim,
                )
                for _ in range(self.num_shared_experts)
            ]
        )

    def _forward_routed_experts(
        self,
        x_flat,
        topk_probs,
        topk_indices,
    ):
        """Run only routed experts that received tokens."""

        routed_out = torch.zeros_like(x_flat)

        unique_experts = torch.unique(topk_indices)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())

            # [tokens, top_k]
            expert_mask = topk_indices == expert_id

            # Tokens for which this expert was selected.
            token_mask = expert_mask.any(dim=-1)

            selected_idx = (
                token_mask
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(
                0,
                selected_idx,
            )

            expert_output = self.experts[expert_id](
                expert_input
            )

            # Find the Top-K slot occupied by this expert.
            selected_mask = expert_mask.index_select(
                0,
                selected_idx,
            )

            slot_indices = (
                selected_mask
                .int()
                .argmax(
                    dim=-1,
                    keepdim=True,
                )
            )

            # Routing probability corresponding to this expert.
            selected_probs = torch.gather(
                topk_probs.index_select(
                    0,
                    selected_idx,
                ),
                dim=-1,
                index=slot_indices,
            ).squeeze(-1)

            # Weighted expert output.
            expert_output = (
                expert_output
                * selected_probs.unsqueeze(-1)
            )

            # Accumulate into the original token positions.
            routed_out.index_add_(
                0,
                selected_idx,
                expert_output,
            )

        return routed_out

    def _forward_shared_experts(self, x_flat):
        """
        Every shared expert is active for every token.

        Their outputs are summed.
        """

        if self.num_shared_experts == 0:
            return torch.zeros_like(x_flat)

        shared_out = torch.zeros_like(x_flat)

        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        return shared_out

    def forward(self, x):
        """
        Args:
            x: [B, S, D]

        Returns:
            [B, S, D]
        """

        batch, seq_len, _ = x.shape

        # [B, S, D] -> [B*S, D]
        x_flat = x.reshape(
            batch * seq_len,
            self.emb_dim,
        )

        # Router logits.
        router_logits = self.gate(x_flat)

        # Select Top-K routed experts.
        topk_logits, topk_indices = torch.topk(
            router_logits,
            self.num_experts_per_tok,
            dim=-1,
        )

        # Routing probabilities only across selected experts.
        topk_probs = torch.softmax(
            topk_logits,
            dim=-1,
        )

        # Routed expert path.
        routed_out = self._forward_routed_experts(
            x_flat,
            topk_probs,
            topk_indices,
        )

        # Shared expert path.
        shared_out = self._forward_shared_experts(
            x_flat
        )

        # Combine both paths.
        output = (self.routed_scaling_factor*routed_out + shared_out)

        # [B*S, D] -> [B, S, D]
        return output.reshape(
            batch,
            seq_len,
            self.emb_dim,
        )


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn = MultiLatentAttention(
            d_model=cfg["emb_dim"],
            n_heads=cfg["num_heads"],
            q_lora_rank=cfg["q_lora_rank"],
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            attention_dropout=cfg["attention_dropout"],
            rms_norm_eps=cfg["rms_norm_eps"],
            llama_4_scaling_beta=cfg["rope_scaling"]["llama_4_scaling_beta"],
            original_max_position_embeddings=cfg["rope_scaling"]["original_max_position_embeddings"],
        )

        self.layer1 = RMSNorm(cfg["emb_dim"],eps=cfg['rms_norm_eps'])
        self.layer2 = RMSNorm(cfg["emb_dim"],eps=cfg['rms_norm_eps'])

        # The first first_k_dense_replace layers use a dense MLP (SwiGLUExpert is a plain
        # SwiGLU feed-forward), every later layer uses the mixture of experts
        if layer_idx < cfg["first_k_dense_replace"]:
            self.ffn = SwiGLUExpert(cfg["emb_dim"], cfg["dense_hidden_dim"])
        else:
            self.ffn = MoEFeedForward(cfg)


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

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_layers"])]
        )

        # One RoPE table shared by all layers
        cos, sin = compute_rope_params(
            head_dim=cfg["qk_rope_head_dim"],
            cfg=cfg
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.out_norm_layer = RMSNorm(cfg["emb_dim"],eps=cfg['rms_norm_eps'])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        batch_size, num_tokens = idx.shape

        if num_tokens > self.cos.size(0):
            raise ValueError(
                f"Sequence length {num_tokens} exceeds "
                f"context_length {self.cos.size(0)}"
            )

        x = self.tok_emb(idx)

        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.out_norm_layer(x)
        logits = self.out_proj(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]


        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  
    return idx


def main():
    # Values come from this folder's params.json (Mistral's native format)
    cfg = {
            "vocab_size": 131072,
            "emb_dim": 7168,
            "hidden_dim": 4096,             # params.json moe.expert_hidden_dim
            "dense_hidden_dim": 16384,      # params.json hidden_dim: MLP size of the dense layers
            "first_k_dense_replace": 3,
            "context_length": 294912,
            "num_heads": 128,
            "num_layers": 61,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "rope_theta": 10000.0,
            "num_experts": 128,
            "num_shared_experts": 1,
            "num_experts_per_tok": 4,
            "attention_dropout": 0.0,
            "shared_expert_hidden_dim": 4096,
            "rms_norm_eps": 1e-06,
            "routed_scaling_factor": 1.0,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 36.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1
            }
        }

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()

    input_text = "Hello, how are you?"

    tokenizer_path = "tokenizer.json"

    tokenizer = Tokenizer.from_file(tokenizer_path)


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
        context_size=cfg["context_length"],
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
