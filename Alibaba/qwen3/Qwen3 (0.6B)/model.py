import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length, stride):
        
        self.input_ids = []
        self.target_ids = []
        
        enc = tokenizer.encode(data)
        token_ids = enc.ids
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
        

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
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
    
    dataset = CustomDataset(
                data, 
                tokenizer,
                context_size,
                stride
            )
    
    dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers
            )
    
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
            context_length, 
            dropout, d_in,
            d_out,
            bias_qkv,
            head_dim,
        ):
        
        
        super().__init__()
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        
        self.num_heads = num_heads
        # Qwen3 sets the head size in the config: it is not hidden_size / num_heads
        self.head_dim = head_dim
        self.kv_head = kv_head
        
        self.d_out = d_out
        
        self.W_q = nn.Linear(d_in, num_heads * self.head_dim, bias=bias_qkv)
        self.W_k = nn.Linear(d_in, kv_head * self.head_dim, bias=bias_qkv)
        self.W_v = nn.Linear(d_in, kv_head * self.head_dim, bias=bias_qkv)
        
        self.q_norm = RMSNorm(
            self.head_dim
        )
        
        self.k_norm = RMSNorm(
            self.head_dim
        )
        
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_out, bias=bias_qkv)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),diagonal=1).bool())
        
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
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        
        groups = self.num_heads // self.kv_head
        
        K = K.repeat_interleave(groups,dim=1)
        V = V.repeat_interleave(groups,dim=1)
        
        attention_score = Q @ K.transpose(3,2)
        
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attention_score = attention_score.masked_fill(mask_bool,float('-inf'))
        
        attention_weights = torch.softmax(attention_score / K.shape[-1] ** 0.5,dim=-1)
        
        context_vector = attention_weights @ V
        
        context_vector = context_vector.transpose(1, 2)
        
        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.num_heads * self.head_dim)
    
        context_vector = self.out_proj(context_vector)
        
        return context_vector
    
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
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

    def __init__(self, emb_dim, hidden_dim=3072):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):

        gate = F.silu(self.gate_proj(x))

        value = self.up_proj(x)

        x = gate * value

        return self.down_proj(x)
    
    
class TransformerBlock(nn.Module):  
    def __init__(self,cfg):
        super().__init__()
        
        self.attn = GroupedQueryAttention(
            num_heads = cfg['num_heads'],
            kv_head = cfg['kv_head'],
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            dropout = cfg['dropout'],
            bias_qkv = cfg['bias_qkv'],
            head_dim = cfg['head_dim']
        )
        
        self.layer1 = RMSNorm(cfg['emb_dim'])
        self.layer2 = RMSNorm(cfg['emb_dim'])
        
        self.ffn = FeedForward(cfg['emb_dim'])
        
        self.attn_dropout = nn.Dropout(cfg['dropout'])
        
    def forward(self,x,cos,sin):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x,cos,sin)
        x = self.attn_dropout(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = self.attn_dropout(x)
        x = x + shortcut
        
        return x
    
    
class Model(nn.Module):
    
    def __init__(self,cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
        self.out_norm_layer = RMSNorm(cfg["emb_dim"])
        self.out_proj = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.out_proj.weight = self.tok_emb.weight

        # One RoPE table shared by every layer
        cos, sin = compute_rope_params(
            cfg["head_dim"], cfg["rope_theta"], cfg["context_length"]
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
        
        x = token_embedding
        
        x = self.dropout(x)
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
        "vocab_size": 151936,
        "emb_dim": 1024,
        "hidden_dim" : 3072,
        "context_length": 40960,
        "num_heads": 16,
        "kv_head": 8,
        "head_dim": 128,
        "num_layers": 28,
        "dropout": 0.0,
        "bias_qkv": False,
        "rope_theta": 1000000
    }
    
    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()
    
    input_text = "Hello, how are you?"
    
    tokenizer = Tokenizer.from_file('tokenizer.json')
    
    print("Tokenizer vocab size:", tokenizer.get_vocab_size())
    print("Model vocab size:", cfg["vocab_size"])

    if tokenizer.get_vocab_size() != cfg["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.get_vocab_size()}) "
            f"does not match model vocab size ({cfg['vocab_size']})."
        )

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