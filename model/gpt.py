import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        # 1. Token embeddings: maps each token ID → dense vector of size model_dim
        #    e.g. token 42 → [0.3, -0.1, 0.8, ...]  shape: (vocab_size, model_dim)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)

        # 2. Position embeddings: learned vectors for each position (0, 1, ..., context_length-1)
        #    Unlike sinusoidal, these are learned during training
        #    shape: (context_length, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)

        # 3. Stack N transformer blocks sequentially
        #    Each block: MultiHeadAttention + FFN with residual connections + LayerNorm
        self.blocks = nn.Sequential(
            *[self.TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)]
        )

        # 4. Final LayerNorm after all blocks (stabilizes output before projection)
        self.final_norm = nn.LayerNorm(model_dim)

        # 5. Vocabulary projection: maps model_dim → vocab_size (raw logits, no softmax)
        #    During training: cross_entropy applies softmax internally
        #    During inference: apply softmax yourself to sample next token
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)

        # 1. Add token embeddings + position embeddings (use torch.arange for positions)
        # 2. Pass through transformer blocks
        # 3. Apply final LayerNorm, then project to vocab_size
        # 4. Return logits rounded to 4 decimal places (no softmax)

        # context shape: (B, T) — batch of token ID sequences
        # T must be <= context_length

        # Step 1: Token embeddings — look up each token ID
        # (B, T) → (B, T, model_dim)
        token_emb = self.token_embedding(context)

        # Step 1b: Position embeddings — create [0, 1, 2, ..., T-1] for each sequence
        # torch.arange(T) → [0, 1, ..., T-1], then embed each position
        # (T,) → (T, model_dim), broadcasts over batch dim automatically
        T = context.shape[1]
        positions = torch.arange(T, device=context.device)
        pos_emb = self.position_embedding(positions)  # (T, model_dim)

        # Add token + position embeddings — gives each token its meaning AND location
        x = token_emb + pos_emb  # (B, T, model_dim)

        # Step 2: Pass through all N transformer blocks
        # Each block: tokens communicate via attention, then process independently via FFN
        x = self.blocks(x)  # (B, T, model_dim)

        # Step 3: Final LayerNorm to stabilize before projection
        x = self.final_norm(x)  # (B, T, model_dim)

        # Step 4: Project to vocab size → logits for every possible next token
        # At position t, logits[t] = scores for what token comes at position t+1
        logits = self.vocab_projection(x)  # (B, T, vocab_size)

        return torch.round(logits, decimals=4)

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
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
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.output_proj(concatenated)
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
            return embedded
