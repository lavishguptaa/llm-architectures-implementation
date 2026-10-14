import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeAttention(nn.Module):
    """Multiplicative (Luong general) attention: score = (W q)^T k."""

    def __init__(self, query_dim, key_dim):
        super().__init__()

        self.query_projection = nn.Linear(
            query_dim,
            key_dim,
            bias=False
        )

    def forward(self, query, keys, values):
        # query: [batch, query_dim]
        # keys: [batch, seq_len, key_dim]
        # values: [batch, seq_len, value_dim]

        query_proj = self.query_projection(query).unsqueeze(1)

        scores = torch.bmm(
            query_proj,
            keys.transpose(1, 2)
        ).squeeze(1)

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            values
        ).squeeze(1)

        return context, attention_weights


if __name__ == "__main__":
    batch_size = 2
    query_dim = 32
    key_dim = 64
    value_dim = 48
    seq_len = 6

    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)

    attention = MultiplicativeAttention(query_dim, key_dim)

    context, attention_weights = attention(query, keys, values)

    print("Query shape:", query.shape)
    print("Context shape:", context.shape)
    print("Attention weights shape:", attention_weights.shape)
