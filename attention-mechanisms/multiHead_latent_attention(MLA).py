import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.latent_dim = latent_dim

        # Query projection
        self.query_projection = nn.Linear(
            d_model,
            d_model
        )

        # Compress input into a latent representation
        self.latent_projection = nn.Linear(
            d_model,
            latent_dim
        )

        # Expand latent representation into K and V
        self.key_projection = nn.Linear(
            latent_dim,
            d_model
        )

        self.value_projection = nn.Linear(
            latent_dim,
            d_model
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        batch_size, seq_len, _ = x.shape

        # Query
        Q = self.query_projection(x)

        # Compress into latent representation
        latent = self.latent_projection(x)

        # Reconstruct K and V from latent representation
        K = self.key_projection(latent)
        V = self.value_projection(latent)

        # Q: [B, S, D] -> [B, H, S, D_head]
        Q = Q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head
        ).transpose(1, 2)

        # K: [B, S, D] -> [B, H, S, D_head]
        K = K.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head
        ).transpose(1, 2)

        # V: [B, S, D] -> [B, H, S, D_head]
        V = V.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head
        ).transpose(1, 2)

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
    latent_dim = 16

    x = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    attention = MultiLatentAttention(
        d_model=d_model,
        n_heads=n_heads,
        latent_dim=latent_dim
    )

    output, attention_weights = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)