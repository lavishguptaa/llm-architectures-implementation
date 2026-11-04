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


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weight * x + self.bias
    

def compute_rope_params(rotary_dim, theta_base, context_length):

    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin, start_pos=0):

    # Partial RoPE: only the first rotary_dim dims of each head are rotated,
    # the remaining dims pass through unchanged (Phi: partial_rotary_factor).
    batch_size, num_heads, seq_len, head_dim = x.shape

    rotary_dim = cos.shape[-1]

    # start_pos is the number of tokens already in the KV cache, so new
    # tokens get their absolute positions instead of restarting at 0
    cos = cos[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[start_pos:start_pos + seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    # Half-split pairing (dimension i with i + rotary_dim / 2), as in Hugging Face checkpoints
    x1 = x_rot[..., :rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    x_rot = (x_rot * cos) + (rotated * sin)

    return torch.cat([x_rot, x_pass], dim=-1)


class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_heads,d_in,d_out,context_length,attention_dropout,bias_qkv=False):
        super().__init__()
        
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_in // num_heads
        self.d_out = d_out
        
        self.W_q = nn.Linear(d_in,d_out,bias=bias_qkv)
        self.W_k = nn.Linear(d_in,d_out,bias=bias_qkv)
        self.W_v = nn.Linear(d_in,d_out,bias=bias_qkv)
        
        
        self.out_proj = nn.Linear(d_out,d_out)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1).bool())
        
    def forward(self,x,cos,sin,layer_cache=None):
        # layer_cache: (K, V) from earlier steps, each [batch, kv_heads, past_len, head_dim].
        # Only the key/value heads are cached.
        batch_size,num_tokens,d_in = x.shape

        past_len = 0 if layer_cache is None else layer_cache[0].size(2)
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        K = K.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        V = V.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        Q = apply_rope(Q, cos, sin, start_pos=past_len)
        K = apply_rope(K, cos, sin, start_pos=past_len)

        # Keys are cached after RoPE, so they are never recomputed
        if layer_cache is not None:
            K = torch.cat([layer_cache[0], K], dim=2)
            V = torch.cat([layer_cache[1], V], dim=2)

        new_cache = (K, V)
        
        attention_scores = Q@K.transpose(2,3)
        
        mask_bool = self.mask.bool()[past_len:past_len + num_tokens, :past_len + num_tokens]
        
        attention_scores = attention_scores.masked_fill(mask_bool, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores / K.shape[-1]**0.5, dim=-1)
        attention_weights =  self.attention_dropout(attention_weights)
        
        context_vector = (attention_weights @ V).transpose(1,2)
        
        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.d_out)
        
        context_vector = self.out_proj(context_vector)
        
        return context_vector, new_cache
    
    

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)
    
    
class TransformerBlock(nn.Module):  
    def __init__(self,cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            num_heads = cfg['num_heads'],
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            attention_dropout = cfg['attention_dropout'],
            bias_qkv = cfg['bias_qkv']
        )
        
        self.layer1 = LayerNorm(
            cfg["emb_dim"],
            cfg["layer_norm_eps"]
        )

        self.layer2 = LayerNorm(
            cfg["emb_dim"],
            cfg["layer_norm_eps"]
        )
        
        self.ffn = FeedForward(cfg['emb_dim'],cfg['intermediate_size'])
        
        
    def forward(self,x,cos,sin,layer_cache=None):
        shortcut = x
        x = self.layer1(x)
        x, new_cache = self.attn(x, cos, sin, layer_cache)
        x = x + shortcut
        
        shortcut = x
        x = self.layer2(x)
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
        
        self.out_norm_layer = LayerNorm(
            cfg["emb_dim"],
            cfg["layer_norm_eps"]
        )

        # One RoPE table shared by every layer; only the first rotary_dim dims of each head are rotated
        head_dim = cfg["emb_dim"] // cfg["num_heads"]
        rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
        cos, sin = compute_rope_params(
            rotary_dim, cfg["rope_theta"], cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.out_proj = nn.Linear(cfg["emb_dim"],cfg["vocab_size"])
        
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
        "vocab_size": 51200,
        "emb_dim": 2048,
        "intermediate_size": 8192,
        "context_length": 2048,
        "num_heads": 32,
        "num_layers": 24,
        "bias_qkv": False,
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "hidden_act": "gelu_new"
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