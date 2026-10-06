import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeAttention(nn.Module):
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