import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
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