import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)

        # Single shared K and V head
        self.key_projection = nn.Linear(d_model, self.d_head)
        self.value_projection = nn.Linear(d_model, self.d_head)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        batch_size, seq_len, d_model = x.shape

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

        # K, V: [B, S, D_head] -> [B, 1, S, D_head]
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        # Shared K and V are broadcast across all query heads
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
            d_model
        )

        output = self.output_projection(output)

        return output, attention_weights


if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 8

    x = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    attention = MultiQueryAttention(
        d_model=d_model,
        n_heads=n_heads
    )

    output, attention_weights = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)