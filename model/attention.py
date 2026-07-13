import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded shape: (B, T, embedding_dim)
        # B = batch size, T = number of tokens in the sequence

        # Step 1: project each token into Key, Query, Value spaces
        # nn.Linear applies: output = input @ W.T
        # (B, T, embedding_dim) → (B, T, attention_dim)
        K = self.key(embedded)    # shape: (B, T, attention_dim)
        Q = self.query(embedded)  # shape: (B, T, attention_dim)
        V = self.value(embedded)  # shape: (B, T, attention_dim)

        # Step 2: compute raw attention scores = Q @ Kᵀ / sqrt(attention_dim)
        # K.transpose(-2, -1) swaps the last two dims:
        #   -2 = second-to-last dim (T), -1 = last dim (attention_dim)
        #   (B, T, attention_dim) → (B, attention_dim, T)
        # matmul: (B, T, attention_dim) @ (B, attention_dim, T) = (B, T, T)
        # cell [b, i, j] = how much token i attends to token j in batch b
        attention_scores = (Q @ K.transpose(-2, -1)) / (attention_dim ** 0.5)
        # shape: (B, T, T)

        # Step 3: causal mask — token at position i can only see positions 0..i
        # shape[-2] = T (rows = query positions)
        # shape[-1] = T (cols = key positions)
        # torch.tril keeps lower triangle (past + present), zeros upper triangle (future)
        mask = torch.tril(torch.ones(attention_scores.shape[-2], attention_scores.shape[-1]))
        # mask shape: (T, T) — broadcasts over batch dim B automatically

        # fill future positions (mask==0) with -inf → softmax turns them into 0.0
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # shape: (B, T, T)

        # Step 4: softmax over dim=-1 (last dim = key/col axis)
        # each row (one query token) sums to 1.0 → attention weights
        attention_scores = torch.softmax(attention_scores, dim=-1)
        # shape: (B, T, T)

        # Step 5: weighted sum of Values
        # (B, T, T) @ (B, T, attention_dim) = (B, T, attention_dim)
        # each output token = blend of all V vectors weighted by attention scores
        return torch.round(attention_scores @ V, decimals=4)
        # shape: (B, T, attention_dim)
        

