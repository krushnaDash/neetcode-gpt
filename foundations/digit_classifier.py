import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.model = nn.Sequential(
            # Input feature 784, output feature 512 (number of neurons in hidden layer)
            # Example x = [0.0, 0.9, 0.1, ..., 0.5]   # 784 pixel values
            # Each of the 512 neurons does:
            # neuron_1 = w1[0]*x[0] + w1[1]*x[1] + ... + w1[783]*x[783] + b1
            # neuron_2 = w2[0]*x[0] + w2[1]*x[1] + ... + w2[783]*x[783] + b2
            # ...
            # neuron_512 = ...
            # output = [neuron_1, neuron_2, ..., neuron_512]   # shape (1, 512)
            # The weights are learned during training via backpropagation — they start random (due to torch.manual_seed(0)) 
            # and gradually adjust to recognize digit patterns, pyTorch uses Kaiming Uniform initialization:
            # W ~ Uniform(-1/√fan_in, +1/√fan_in) W ~ Uniform(-1/√784, +1/√784)  =  Uniform(-0.0357, +0.0357)
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Input feature 512, output feature 10 (number of classes)
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x= self.model(images)
        return torch.round(x, decimals=4)

