import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)
        X = torch.zeros((batch_size, context_length), dtype=torch.int64)
        Y = torch.zeros((batch_size, context_length), dtype=torch.int64)
        for i in range(batch_size):
            start_idx = torch.randint(0, len(data) - context_length, (1,)).item()
            X[i] = data[start_idx:start_idx + context_length]
            Y[i] = data[start_idx + 1:start_idx + context_length + 1]
        return X, Y
    