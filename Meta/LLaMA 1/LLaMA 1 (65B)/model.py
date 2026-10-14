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


class Rope(nn.Module):

    def __init__(self, head_dim, seq_len, theta):
        super().__init__()

        assert head_dim % 2 == 0, "head_dim must be even"

        self.head_dim = head_dim
        self.max_seq_length = seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))


        positions = torch.arange(seq_len, dtype=torch.float32)

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer(
                "cos_cached", 
                freqs.cos(),
                persistent=False
        )
        
        self.register_buffer(
                "sin_cached",
                freqs.sin(),
                persistent=False
        )

    def forward(self, x, position_ids=None):
        seq_len = x.size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=x.device,
            )
        else:
            position_ids = position_ids.to(x.device)

        if position_ids.max().item() >= self.max_seq_length:
            raise ValueError(
                f"Position {position_ids.max().item()} exceeds "
                f"RoPE max_seq_len={self.max_seq_length}"
            )

        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        x_rotated = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)

        return x_rotated

class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_heads,d_in,d_out,context_length,attention_dropout,rope_base,bias_qkv=False):
        super().__init__()
        
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_in // num_heads
        self.d_out = d_out
        
        self.W_q = nn.Linear(d_in,d_out,bias=bias_qkv)
        self.W_k = nn.Linear(d_in,d_out,bias=bias_qkv)
        self.W_v = nn.Linear(d_in,d_out,bias=bias_qkv)
        
        
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.rope = Rope(
            head_dim=self.head_dim,
            seq_len=context_length,
            theta=rope_base
        )
        
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1).bool())
        
    def forward(self,x):
        batch_size,num_tokens,d_in = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        K = K.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        V = V.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        Q = self.rope(Q)
        K = self.rope(K)
        
        attention_scores = Q@K.transpose(2,3)
        
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        
        attention_scores = attention_scores.masked_fill(mask_bool, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores / K.shape[-1]**0.5, dim=-1)
        attention_weights =  self.attention_dropout(attention_weights)
        
        context_vector = (attention_weights @ V).transpose(1,2)
        
        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.d_out)
        
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
        
        self.attn = MultiHeadAttention(
            num_heads = cfg['num_heads'],
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            attention_dropout = cfg['attention_dropout'],
            bias_qkv = cfg['bias_qkv'],
            rope_base = cfg['rope_theta']
        )
        
        self.layer1 = RMSNorm(cfg['emb_dim'],cfg['rms_norm_eps'])
        self.layer2 = RMSNorm(cfg['emb_dim'],cfg['rms_norm_eps'])
        
        self.ffn = FeedForward(cfg['emb_dim'],cfg['intermediate_size'])
        
        
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
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
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
        "vocab_size": 32000,
        "emb_dim": 8192,
        "intermediate_size": 22016,
        "context_length": 2048,
        "num_heads": 64,
        "num_layers": 80,
        "bias_qkv": False,
        "rms_norm_eps":1e-05,
        "attention_dropout": 0.0,
        "rope_theta": 10000.0
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