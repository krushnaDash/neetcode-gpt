import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stat=[]
        #PyTorch normally tracks every operation to compute gradients (for training). 
        # Here we're just observing — not training — so we tell PyTorch "don't waste memory tracking gradients."
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = round(float(torch.mean(x).item()), 4)
                    std  = round(float(torch.std(x).item()), 4)
                    # a neuron is dead if ALL samples in the batch have a non-positive output
                    if x.dim() >= 2:
                        dead_fraction = round(float(((x <= 0).all(dim=0)).float().mean().item()), 4)
                    else:
                        dead_fraction = round(float((x <= 0).float().mean().item()), 4)
                    stat.append({"mean": mean, "std": std, "dead_fraction": dead_fraction})
        return stat

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad() # Reset the Gradients
        output = model(x)  # forward pass
        loss = nn.MSELoss()(output, y)
        loss.backward() # backward pass with grad computation
        stats = []
        
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = float(torch.mean(grad).item())
                std = float(torch.std(grad).item())
                norm = float(torch.norm(grad).item())
                stats.append({"mean": round(mean, 4), "std": round(std, 4), "norm": round(norm, 4)})
        return stats
        

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'
        # Check if the last layer has vanishing gradients
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'    
        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10.0:
                return 'exploding_gradients'
            
        return 'healthy'
