#!/usr/bin/env ipython

import torch.nn as nn


class VectorPredictor(nn.Module):
    def __init__(self):
        super(VectorPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),  # Input is 3D vector, output 64 features
            nn.ReLU(),
            nn.Linear(64, 3)  # Output is 3D vector
        )
        self.model1 = nn.Sequential(
            nn.Linear(3, 3),  # Input is 3D vector, output 64 features
            # nn.ReLU(),
            # nn.Linear(64, 3)  # Output is 3D vector
        )

    def forward(self, x):
        return self.model1(x)
