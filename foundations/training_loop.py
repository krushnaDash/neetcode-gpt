import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w=np.zeros(X.shape[1])
        b=0.0
        n=len(y)

        for _ in range(epochs):
            # Forward pass
            y_hat=X@w+b
            # we just need the error, not the full loss for gradient computation
            error=y_hat-y 

            # Compute gradients of MSE loss
            dw= (2.0 / n) * (X.T @ error)
            # X is (n_samples, n_features) and error is (n_samples,). We need dw to be (n_features,) — one gradient per weight.
            # X.T flips the matrix from (n_samples, n_features) → (n_features, n_samples) so the matrix multiply works out to the right shape.
            db= (2.0 / n) * np.sum(error)
            
            w=w-lr*dw
            b=b-lr*db
        
        return (np.round(w, 5), round(b, 5))