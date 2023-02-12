"""
This file contains a NN module to define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=128):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_dim)

        self.swish = lambda x: x * torch.sigmoid(x)
        self.initialize_weights()  # orthogonal initialization

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        activation1 = self.swish(self.layer1(state))
        activation2 = self.swish(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
