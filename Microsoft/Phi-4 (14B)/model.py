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

    # Half-split pairing (dimension i with i + head_dim / 2), as in Hugging Face checkpoints
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self,
            num_heads,
            kv_head, 
            attention_dropout,
            d_in,
            d_out, 
            bias_qkv=False,
        ):
        
        super().__init__()
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_in // num_heads
        self.kv_head = kv_head
        
        self.d_in = d_in
        self.d_out = d_out
        
        self.attention_dropout = attention_dropout
        
        self.W_q = nn.Linear(d_in, num_heads * self.head_dim, bias=bias_qkv)
        self.W_k = nn.Linear(d_in, kv_head * self.head_dim, bias=bias_qkv)
        self.W_v = nn.Linear(d_in, kv_head * self.head_dim, bias=bias_qkv)
        
        self.out_proj = nn.Linear(d_in, d_out,bias=bias_qkv)
        
    def forward(self,x,cos,sin):
        batch_size,num_tokens,d_in = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        K = K.view(batch_size,num_tokens,self.kv_head,self.head_dim)
        V = V.view(batch_size,num_tokens,self.kv_head,self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        
        
        context = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=(
                   self.attention_dropout
                    if self.training
                    else 0.0
                ),
            is_causal=True,
            enable_gqa=True,
        )
        
        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            num_tokens,
            self.d_out,
        )


        return self.out_proj(context)
    
class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim)
        self.up_proj = nn.Linear(emb_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        """Apply the gated feed-forward transformation."""
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        x = gate * value

        return self.down_proj(x)
    
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
    
class TransformerBlock(nn.Module):  
    def __init__(self,cfg):
        super().__init__()
        
        self.attn = GroupedQueryAttention(
            num_heads = cfg['num_heads'],
            kv_head = cfg['kv_head'],
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            bias_qkv = cfg['bias_qkv'],
           attention_dropout=cfg["attention_dropout"]
        )
        
        self.layer1 = RMSNorm(cfg['emb_dim'],eps=cfg["rms_norm_eps"])
        self.layer2 = RMSNorm(cfg['emb_dim'],eps=cfg["rms_norm_eps"])
        
        self.ffn = FeedForward(cfg['emb_dim'],hidden_dim = cfg['intermediate_size'])
        
        self.resid_dropout = nn.Dropout(cfg["resid_dropout"])
        
    def forward(self,x,cos,sin):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x,cos,sin)
        x = self.resid_dropout(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = self.resid_dropout(x)
        x = x + shortcut
        
        return x
    
class Model(nn.Module):
    
    def __init__(self,cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.embd_dropout = nn.Dropout(cfg["embd_dropout"])
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
        self.out_norm_layer = RMSNorm(cfg["emb_dim"],eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["emb_dim"],cfg["vocab_size"])

        # One RoPE table shared by every layer
        head_dim = cfg["emb_dim"] // cfg["num_heads"]
        cos, sin = compute_rope_params(
            head_dim, cfg["rope_base"], cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self,idx):
        batch_size,num_tokens = idx.shape
        
        if idx.size(1) > self.cos.size(0):
            raise ValueError(
                f"Sequence length {idx.size(1)} exceeds "
                f"context_length {self.cos.size(0)}"
            )

        token_embedding = self.tok_emb(idx)
        x = self.embd_dropout(token_embedding)
        
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
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
        "vocab_size": 100352,
        "emb_dim": 5120,
        "intermediate_size" : 17920,
        "context_length": 16384,
        "num_heads": 40,
        "kv_head": 10,
        "num_layers": 40,
        "attention_dropout": 0.0,
        "resid_dropout": 0.0,
        "embd_dropout": 0.0,
        "bias_qkv": False,
        "rope_base": 250000,
        "rms_norm_eps": 1e-5,
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
            context_size=cfg["context_length"]
        )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
    


if __name__ == "__main__":
    main()