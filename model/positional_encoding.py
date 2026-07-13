import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.

        # Step 1: positions [0, 1, 2, ..., seq_len-1] as a column vector
        # shape: (seq_len, 1)  e.g. seq_len=3 → [[0], [1], [2]]
        pos = np.arange(seq_len).reshape(-1, 1) 
        # -1 means auto-determine the dimension for row

        # Step 2: pair indices [0, 1, ..., d_model/2 - 1] as a row vector
        # shape: (1, d_model//2)  e.g. d_model=4 → [[0, 1]]
        # // integer division to get number of pairs
        i = np.arange(d_model // 2).reshape(1, -1) # -1 means auto-determine the dimension

        # Step 3: divisor controls wave speed — grows with i
        # i=0 → 10000^0 = 1.0 (fast), i=1 → 10000^0.5 = 100.0 (slow)
        # shape: (1, d_model//2)
        divisor = np.power(10000, (2 * i) / d_model)

        # Step 4: angles = pos / divisor
        # Broadcasting: (seq_len, 1) / (1, d_model//2) → (seq_len, d_model//2)
        # each row = one position, each col = one wave pair's angle
        angles = pos / divisor

        # Step 5: fill the PE matrix — sin for even cols, cos for odd cols
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angles)  # even columns: 0, 2, 4, ...
        pe[:, 1::2] = np.cos(angles)  # odd  columns: 1, 3, 5, ...

        return np.round(pe, 5)
        
