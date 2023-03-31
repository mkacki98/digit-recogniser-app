import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch.nn as nn
import torch
import torch.multiprocessing

from src.dataset.create_dataset import create_dataloaders
from src.modelling.architectures import MLP, CNN, NeuromorphicClassifier
from src.utils.utils import load_configs, display_training_examples, get_device
from src.utils.utils_train import get_pretrained_synapses, train_validate


def main():
    device = get_device()
    config = load_configs()

    train_loader, val_loader, _ = create_dataloaders(config.batch_size)
    criterion = nn.CrossEntropyLoss()

    if config.model == "cnn":
        model = CNN()
    elif config.model == "mlp":
        model = MLP()
    else:
        synapses = get_pretrained_synapses(train_loader, config, from_pretrained=True)
        model = NeuromorphicClassifier(synapses)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    display_training_examples()
    train_validate(
        model, config, train_loader, val_loader, optimizer, criterion, device
    )


if __name__ == "__main__":
    main()
