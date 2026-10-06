import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        assert n_heads % n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)

        self.key_projection = nn.Linear(
            d_model,
            n_kv_heads * self.d_head
        )

        self.value_projection = nn.Linear(
            d_model,
            n_kv_heads * self.d_head
        )

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        batch_size, seq_len, _ = x.shape

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # Q: [B, S, D] -> [B, H, S, D_head]
        Q = Q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head
        ).transpose(1, 2)

        # K, V: [B, S, H_kv * D_head]
        # -> [B, H_kv, S, D_head]
        K = K.view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.d_head
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.d_head
        ).transpose(1, 2)

        # Number of query heads sharing each KV head
        groups = self.n_heads // self.n_kv_heads

        # Repeat each KV head for its corresponding query group
        K = K.repeat_interleave(groups, dim=1)
        V = V.repeat_interleave(groups, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / (self.d_head ** 0.5)

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        # [B, H, S, D_head] -> [B, S, D]
        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size,
            seq_len,
            self.n_heads * self.d_head
        )

        output = self.output_projection(output)

        return output, attention_weights


if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 8
    n_kv_heads = 2

    x = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    attention = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads
    )

    output, attention_weights = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)