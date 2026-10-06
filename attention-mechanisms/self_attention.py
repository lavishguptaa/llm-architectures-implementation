import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.scale = d_model ** 0.5

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        scores = torch.bmm(
            Q,
            K.transpose(1, 2)
        ) / self.scale

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.bmm(
            attention_weights,
            V
        )

        return output, attention_weights


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model)

    attention = ScaledDotProductAttention(d_model)

    output, attention_weights = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)