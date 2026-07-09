import numpy as np
from typing import List


class Solution:
     def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # compute the forward pass
        x= np.array(x)
        W1= np.array(W1)
        b1= np.array(b1)
        W2= np.array(W2)
        b2= np.array(b2)
        y_true= np.array(y_true)

        z1= np.dot(W1, x) + b1
        a1= np.maximum(0, z1)
        z2= np.dot(W2, a1) + b2
        predictions= z2
        loss= np.mean((predictions - y_true) ** 2)

        # compute the backward pass
        
        dz2= 2 * (predictions - y_true) / len(y_true) # dL/dz2: 2x from d(x^2)=2x, /N for mean
        dw2= np.outer(dz2, a1) # dL/dW2 (gradient w.r.t. W2)
        db2=dz2 # dL/db2 (gradient w.r.t. b2)
        da1= np.dot(W2.T, dz2) # dL/da1 (gradient w.r.t. a1)
        dz1= da1 * (z1 > 0) # dL/dz1 (gradient w.r.t. z1)
        dw1= np.outer(dz1, x) # dL/dW1 (gradient w.r.t. W1)
        db1=dz1 # dL/db1 (gradient w.r.t. b1)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }