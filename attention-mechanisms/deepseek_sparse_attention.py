import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        index_heads: int = 4,
        index_head_dim: int = 64,
    ):
        super().__init__()

        self.index_heads = index_heads
        self.index_head_dim = index_head_dim

        self.query_proj = nn.Linear(
            hidden_size,
            index_heads * index_head_dim,
            bias=False,
        )
        self.key_proj = nn.Linear(
            hidden_size,
            index_head_dim,
            bias=False,
        )
        self.weight_proj = nn.Linear(
            hidden_size,
            index_heads,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.query_proj(x).view(
            batch, seq_len, self.index_heads,
            self.index_head_dim,
        )
        k = self.key_proj(x)

        weights = self.weight_proj(x) / math.sqrt(
            self.index_heads
        )

        # [B, H_index, T_query, T_key]
        logits = torch.einsum(
            "bthd,bsd->bhts", q.float(), k.float()
        ) / math.sqrt(self.index_head_dim)

        # ReLU is applied to each index head before aggregation.
        logits = F.relu(logits)

        # [B, T_query, T_key]
        scores = torch.einsum(
            "bhts,bth->bts", logits, weights.float()
        )

        return scores


class DeepSeekSparseAttention(nn.Module):
    """
    Educational DSA reference implementation.

    q, k, v: [B, H, T, head_dim]
    x:       [B, T, hidden_size]

    This version uses gathered selected tokens. It does not
    implement DeepSeek's MLA, FP8 indexer, or fused CUDA kernels.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        index_heads: int = 4,
        index_head_dim: int = 64,
        topk: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.topk = topk
        self.dropout = dropout

        self.indexer = LightningIndexer(
            hidden_size=hidden_size,
            index_heads=index_heads,
            index_head_dim=index_head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        _, num_heads, _, head_dim = q.shape

        if k.shape != v.shape:
            raise ValueError("K and V must have identical shapes.")

        if q.shape[0] != batch or q.shape[2] != seq_len:
            raise ValueError("Q must match x's batch and sequence.")

        # Step 1: Learned relevance scores.
        scores = self.indexer(x)

        # Step 2: Causal mask; a token cannot select future tokens.
        positions = torch.arange(seq_len, device=x.device)
        causal = positions[None, :] <= positions[:, None]

        scores = scores.masked_fill(
            ~causal.unsqueeze(0), float("-inf")
        )

        # Step 3: Select up to topk available tokens per query.
        k_select = min(self.topk, seq_len)
        selected_scores, indices = torch.topk(
            scores, k=k_select, dim=-1
        )

        # Gather the actual attention K/V entries.
        # indices: [B, T, K]
        gather_indices = indices[:, None, :, :, None].expand(
            batch, num_heads, seq_len, k_select, head_dim
        )

        selected_k = torch.gather(
            k.unsqueeze(2).expand(
                -1, -1, seq_len, -1, -1
            ),
            dim=3,
            index=gather_indices,
        )

        selected_v = torch.gather(
            v.unsqueeze(2).expand(
                -1, -1, seq_len, -1, -1
            ),
            dim=3,
            index=gather_indices,
        )

        # [B, H, T, K]
        q_float = q.float()
        k_float = selected_k.float()

        attn_scores = torch.einsum(
            "bhtd,bhtkd->bhtk", q_float, k_float
        ) / math.sqrt(head_dim)

        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        output = torch.einsum(
            "bhtk,bhtkd->bhtd",
            attn_weights,
            selected_v.float(),
        )

        return output.to(q.dtype)