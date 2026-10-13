import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length, stride):
        
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(data).ids
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
                
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(data, tokenizer, context_size=256, stride=128,
                      batch_size=8, shuffle=True, num_workers=0, drop_last=False):
    
    if tokenizer is None:
        raise ValueError("tokenizer must be provided; load tokenizer.json before creating the dataloader")
    
    dataset = CustomDataset(data, tokenizer, context_size, stride)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    
    return dataloader



def compute_rope_params(head_dim, cfg):
    theta_base = cfg["rope_theta"]
    context_length = cfg["context_length"]

    rope_cfg = cfg.get("rope_scaling")
    attention_factor = 1.0

    assert head_dim % 2 == 0, "head_dim must be even"

    if rope_cfg is not None and rope_cfg.get("rope_type") == "yarn":
        rope_factor = rope_cfg["factor"]
        rope_orig_max = rope_cfg["original_max_position_embeddings"]
        beta_fast = rope_cfg["beta_fast"]
        beta_slow = rope_cfg["beta_slow"]
        attention_factor = rope_cfg.get("attention_factor", 1.0)

        if rope_factor <= 0:
            raise ValueError("rope_scaling factor must be positive")

        def find_correction_dim(num_rotations):
            return (
                head_dim
                * math.log(
                    rope_orig_max / (num_rotations * 2 * math.pi)
                )
                / (2 * math.log(theta_base))
            )

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001

            ramp = (
                torch.arange(dim, dtype=torch.float32) - min_val
            ) / (max_val - min_val)

            return torch.clamp(ramp, 0.0, 1.0)

        pos_freqs = theta_base ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (
            rope_factor * pos_freqs
        )

        low = math.floor(find_correction_dim(beta_fast))
        high = math.ceil(find_correction_dim(beta_slow))

        low = max(low, 0)
        high = min(high, head_dim // 2 - 1)

        ramp = linear_ramp_factor(
            low, high, head_dim // 2
        )

        inv_freq_extrapolation_factor = 1.0 - ramp

        inv_freq = (
            inv_freq_interpolation
            * (1.0 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation
            * inv_freq_extrapolation_factor
        )

    else:
        inv_freq = 1.0 / (
            theta_base ** (
                torch.arange(0, head_dim, 2).float()
                / head_dim
            )
        )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    cos = angles.cos() * attention_factor
    sin = angles.sin() * attention_factor

    return cos, sin

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
    def __init__(self, cfg, attention_type="full_attention"):
        super().__init__()

        num_heads = cfg["num_heads"]
        
        d_in = cfg["emb_dim"]
        d_out = cfg["emb_dim"]

        if d_in % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        if attention_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.num_heads = num_heads
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.d_out = d_out
        self.attention_type = attention_type
        self.sliding_window = cfg["sliding_window"]
        self.attention_dropout_p = cfg["attention_dropout"]

        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=cfg["bias_qkv"])
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=cfg["bias_qkv"])
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=cfg["bias_qkv"])
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_out, bias=False)

        self.cos, self.sin = compute_rope_params(
            head_dim=self.head_dim,
            cfg=cfg
        )
        
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=cfg["rms_norm_eps"]
        )

        self.k_norm = RMSNorm(
            self.head_dim,
            eps=cfg["rms_norm_eps"]
        )

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
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        Q = apply_rope(Q, self.cos, self.sin)
        K = apply_rope(K, self.cos, self.sin)

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
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True
        )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out
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

        x = gate * value

        return self.down_proj(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        layer_types = cfg["layer_types"]

        attention_type = (
            layer_types[layer_idx]
            if layer_types is not None
            else "full_attention"
        )

        self.attn = GroupedQueryAttention(cfg, attention_type)

        self.layer1 = RMSNorm(
            cfg["emb_dim"], cfg["rms_norm_eps"]
        )
        self.layer2 = RMSNorm(
            cfg["emb_dim"], cfg["rms_norm_eps"]
        )

        self.ffn = FeedForward(
            cfg["emb_dim"], cfg["intermediate_size"]
        )
        
        
    def forward(self,x):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = x + shortcut
        
        return x
    
    
class Model(nn.Module):
    
    def __init__(self,cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        
        layer_types = cfg["layer_types"]
        
        if layer_types is not None and len(layer_types) != cfg["num_layers"]:
            raise ValueError("layer_types length must equal num_layers")
        
        blocks = []

        for layer_idx in range(cfg["num_layers"]):
            blocks.append(TransformerBlock(cfg, layer_idx))

        self.blocks = nn.Sequential(*blocks)
        
        self.out_norm_layer = RMSNorm(cfg["emb_dim"],cfg['rms_norm_eps'])
        self.out_proj = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        # self.out_proj.weight = self.tok_emb.weight
        
    def forward(self,idx):
        batch_size,num_tokens = idx.shape
        
        token_embedding = self.tok_emb(idx)
        
        x = token_embedding
        
        x = self.blocks(x)
        x = self.out_norm_layer(x)
        x = self.out_proj(x)
        return x
    
    
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
    cfg = {
    "vocab_size": 256000,
    "emb_dim": 2304,
    "head_dim": 256,
    "intermediate_size": 9216,
    "context_length": 8192,
    "num_heads": 8,
    "num_key_value_heads":4,
    "num_layers": 26,
    "bias_qkv": False,
    "rms_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "rope_scaling": None,
    "layer_types": [
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
    ],
}

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()
    
    input_text = "Hello, how are you?"
    
    tokenizer = Tokenizer.from_file('tokenizer.json')
    
    print("Tokenizer vocab size:", tokenizer.get_vocab_size())
    print("Model vocab size:", cfg["vocab_size"])

    # if tokenizer.get_vocab_size() != cfg["vocab_size"]:
    #     raise ValueError(
    #         f"Tokenizer vocab size ({tokenizer.get_vocab_size()}) "
    #         f"does not match model vocab size ({cfg['vocab_size']})."
    #     )

    encoding = tokenizer.encode(input_text)
    input_ids = encoding.ids

    encoded_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    out = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=cfg["context_length"]
        )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

if __name__ == "__main__":
    main()