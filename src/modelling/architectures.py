import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 784
        output_size = 10

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return F.softmax(out, dim=1)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        output_size = 10

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(16 * (7) * (7), output_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return F.softmax(out, dim=1)


class NeuromorphicClassifier(nn.Module):
    def __init__(self, synapses, n=1):
        super().__init__()

        self.synapses = synapses

        hidden_size = synapses.shape[0]
        output_size = 10

        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, v):
        Wv = F.linear(v, self.synapses, None)
        h = self.linear(Wv)
        return F.log_softmax(h, dim=-1)
