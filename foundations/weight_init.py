import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        w = torch.randn(fan_out, fan_in) * math.sqrt(2.0 / (fan_in + fan_out))
        return torch.round(w, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        w = torch.randn(fan_out, fan_in) * math.sqrt(2.0 / fan_in)
        return torch.round(w, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        # Build layer dims: [input_dim, hidden_dim, hidden_dim, ..., hidden_dim]
        dims = [input_dim] + [hidden_dim] * num_layers

        # Generate ALL weight matrices first before drawing x.
        # Cannot call self.xavier_init() or self.kaiming_init() here because
        # each of those methods calls torch.manual_seed(0) internally, which
        # resets the RNG on every iteration — all layers would get identical
        # weight matrices. Instead we compute std inline so the seed is set
        # once above and torch.randn() advances the RNG naturally each loop.
        weights = []
        for i in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0  # raw N(0,1) — no scaling, activations explode
            weights.append(torch.randn(dims[i + 1], dims[i]) * std)

        # x is drawn AFTER all weights so the RNG sequence matches expectations
        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
        return stds
        
        
      
