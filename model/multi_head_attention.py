import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        
        head_size = attention_dim // num_heads  # each head gets equal slice of attention_dim

        # build each head one by one and collect into a list
        heads_list = []
        for _ in range(num_heads):
            head = self.SingleHeadAttention(embedding_dim, head_size)
            heads_list.append(head)

        # nn.ModuleList so PyTorch tracks all heads as learnable parameters
        self.heads = nn.ModuleList(heads_list)
        self.output_projection = nn.Linear(attention_dim, attention_dim, bias=False)
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # run every head on the same input independently
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(embedded))  # each head: (B, T, head_size)

        # join all head outputs side by side along the feature dim
        # num_heads × (B, T, head_size) → (B, T, attention_dim)
        concatenated = torch.cat(head_outputs, dim=2)

        # mix across heads with learned output projection W_O
        out = self.output_projection(concatenated)  # (B, T, attention_dim)

        return torch.round(out, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
