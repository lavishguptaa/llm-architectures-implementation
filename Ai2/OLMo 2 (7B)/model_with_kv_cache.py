import torch
import torch.nn as nn
import torch.nn.functional as F
import json
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

    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    cos = angles.cos()
    sin = angles.sin()

    return cos, sin


def apply_rope(x, cos, sin, start_pos=0):

    batch_size, num_heads, seq_len, head_dim = x.shape

    # start_pos is the number of tokens already in the KV cache, so new
    # tokens get their absolute positions instead of restarting at 0
    cos = cos[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        num_heads = cfg["num_heads"]
        d_in = cfg["emb_dim"]
        d_out = cfg["emb_dim"]
        
        if d_in % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        
        self.num_heads = num_heads
        self.head_dim = d_in // num_heads
        self.d_out = d_out
        
        self.W_q = nn.Linear(d_in, d_out, bias=cfg["bias_qkv"])
        self.W_k = nn.Linear(d_in, d_out, bias=cfg["bias_qkv"])
        self.W_v = nn.Linear(d_in, d_out, bias=cfg["bias_qkv"])
        
        self.q_norm = RMSNorm(d_out, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(d_out, cfg["rms_norm_eps"])
        
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.attention_dropout_p = cfg["attention_dropout"]
        
        
    def forward(self,x,cos,sin,layer_cache=None):
        # layer_cache: (K, V) from earlier steps, each [batch, heads, past_len, head_dim]
        batch_size,num_tokens,d_in = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        Q = Q.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        K = K.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        V = V.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        past_len = 0 if layer_cache is None else layer_cache[0].size(2)

        Q = apply_rope(Q, cos, sin, past_len)
        K = apply_rope(K, cos, sin, past_len)

        # Keys are cached after QK-norm and RoPE, so they are never recomputed
        if layer_cache is not None:
            K = torch.cat([layer_cache[0], K], dim=2)
            V = torch.cat([layer_cache[1], V], dim=2)

        new_cache = (K, V)

        if past_len == 0:
            attn_mask = None
            is_causal = True
        else:
            # is_causal assumes queries and keys have equal length, so build the
            # mask explicitly: query i (absolute position past_len + i) sees keys 0..past_len + i
            q_pos = torch.arange(past_len, past_len + num_tokens, device=x.device)
            k_pos = torch.arange(K.size(2), device=x.device)
            attn_mask = k_pos[None, :] <= q_pos[:, None]
            is_causal = False

        context_vector = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        context_vector = context_vector.transpose(1,2)

        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.d_out)

        context_vector = self.out_proj(context_vector)

        return context_vector, new_cache
    
           
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
    

class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        """Apply the gated feed-forward transformation."""
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        x = gate * value

        return self.down_proj(x)
    
    
class TransformerBlock(nn.Module):  
    def __init__(self,cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(cfg)
        
        self.pre_attn_norm = RMSNorm(cfg['emb_dim'],cfg['rms_norm_eps'])
        self.pre_ffn_norm = RMSNorm(cfg['emb_dim'],cfg['rms_norm_eps'])
        
        self.ffn = FeedForward(cfg['emb_dim'],cfg['intermediate_size'])
        
        
    def forward(self,x,cos,sin,layer_cache=None):
        shortcut = x
        x = self.pre_attn_norm(x)
        x, new_cache = self.attn(x,cos,sin,layer_cache)
        x = x + shortcut

        shortcut = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, new_cache
    
    
class Model(nn.Module):
    
    def __init__(self,cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["num_heads"],
            cfg=cfg
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.out_norm_layer = RMSNorm(cfg["emb_dim"],cfg['rms_norm_eps'])
        self.out_proj = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        # self.out_proj.weight = self.tok_emb.weight
        
    def forward(self,idx,cache=None,use_cache=False):
        # cache: list with one (K, V) pair per layer, from the previous forward call
        # use_cache=False returns logits only; use_cache=True returns (logits, cache)
        batch_size,num_tokens = idx.shape

        past_len = 0 if cache is None else cache[0][0].size(2)

        if past_len + num_tokens > self.cos.size(0):
            raise ValueError(
                f"Sequence length {past_len + num_tokens} exceeds "
                f"context_length {self.cos.size(0)}"
            )

        token_embedding = self.tok_emb(idx)

        x = token_embedding

        new_cache = []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[layer_idx]
            x, layer_cache = block(x, self.cos, self.sin, layer_cache)
            new_cache.append(layer_cache)
        x = self.out_norm_layer(x)
        x = self.out_proj(x)

        if use_cache:
            return x, new_cache
        return x


def generate_text_simple(model, idx, max_new_tokens, context_size):

    # Unlike the uncached version, the context cannot slide: cropping old tokens
    # would invalidate the cached keys/values and their RoPE positions
    if idx.size(1) + max_new_tokens > context_size:
        raise ValueError(
            "prompt length + max_new_tokens must not exceed context_size "
            "when using the KV cache"
        )

    with torch.no_grad():
        # Prefill: process the whole prompt once and fill the cache
        logits, cache = model(idx, use_cache=True)

        for step in range(max_new_tokens):

            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

            # Decode: feed only the new token; earlier K/V come from the cache
            if step < max_new_tokens - 1:
                logits, cache = model(idx_next, cache=cache, use_cache=True)

    return idx
    
def main():
    cfg = {
        "vocab_size": 100352,
        "emb_dim": 4096,
        "intermediate_size": 11008,
        "context_length": 4096,
        "num_heads": 32,
        "num_layers": 32,
        "bias_qkv": False,
        "rms_norm_eps":1e-06,
        "attention_dropout": 0.0,
        "rope_theta": 500000.0
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