import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        stat=[]
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    # Compute dead fraction for this ReLU layer
                    dead_fraction = float(((x <= 0).all(dim=0)).float().mean().item())
                    # Round to 4 decimals
                    dead_fraction = round(dead_fraction, 4)
                    stat.append(dead_fraction)
        return stat            

        
        

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        # 1. ANY layer > 0.5 → use_leaky_relu
        if any(d > 0.5 for d in dead_fractions):
            return "use_leaky_relu"
        # 2. FIRST layer > 0.3 → reinitialize
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"
        # 3. strictly increasing with depth AND last > 0.1 → reduce_learning_rate
        strictly_increasing = all(dead_fractions[i] < dead_fractions[i+1] for i in range(len(dead_fractions)-1))
        if strictly_increasing and dead_fractions and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        return "healthy"
