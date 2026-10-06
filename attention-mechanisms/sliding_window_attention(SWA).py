import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        causal: bool = False,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.causal = causal

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)

        output = self._sliding_window_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        return self.output_projection(output)

    def _sliding_window_attention(self, Q, K, V):
        batch_size, n_heads, seq_len, _ = Q.shape

        outputs = []

        for start in range(0, seq_len, self.window_size):
            end = min(
                start + self.window_size,
                seq_len,
            )

            Q_window = Q[:, :, start:end, :]

            if self.causal:
                K_start = max(0, start - self.window_size + 1)
                K_end = end
            else:
                K_start = max(0, start - self.window_size)
                K_end = min(
                    seq_len,
                    end + self.window_size,
                )

            K_window = K[:, :, K_start:K_end, :]
            V_window = V[:, :, K_start:K_end, :]

            scores = torch.matmul(
                Q_window,
                K_window.transpose(-2, -1),
            ) / (self.d_head ** 0.5)

            if self.causal:
                q_positions = torch.arange(
                    start,
                    end,
                    device=Q.device,
                )

                k_positions = torch.arange(
                    K_start,
                    K_end,
                    device=Q.device,
                )

                causal_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)

                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0),
                    float("-inf"),
                )

            attention_weights = F.softmax(
                scores,
                dim=-1,
            )

            output = torch.matmul(
                attention_weights,
                V_window,
            )

            outputs.append(output)

        return torch.cat(outputs, dim=2)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    d_model = 512
    n_heads = 8
    window_size = 32

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    attention = SlidingWindowAttention(
        d_model=d_model,
        n_heads=n_heads,
        window_size=window_size,
        causal=True,
    )

    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)