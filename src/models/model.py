import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 784
        output_size = 10

        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return F.softmax(out, dim=1)
