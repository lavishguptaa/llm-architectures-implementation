import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention: score = v^T tanh(W_q q + W_k k)."""

    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()

        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, values):
        # query: [batch, query_dim]
        # keys: [batch, seq_len, key_dim]
        # values: [batch, seq_len, value_dim]

        query_proj = self.query_projection(query).unsqueeze(1)
        key_proj = self.key_projection(keys)

        scores = self.score_projection(
            torch.tanh(query_proj + key_proj)
        ).squeeze(-1)

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
    hidden_dim = 16

    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)

    attention = AdditiveAttention(query_dim, key_dim, hidden_dim)

    context, attention_weights = attention(query, keys, values)

    print("Query shape:", query.shape)
    print("Context shape:", context.shape)
    print("Attention weights shape:", attention_weights.shape)
