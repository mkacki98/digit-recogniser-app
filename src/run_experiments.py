import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch.nn as nn
import torch
import torch.multiprocessing

from src.modelling.architectures import MLP, CNN
from src.modelling.train import train_validate
from src.dataset.create_dataset import create_dataloaders
from src.utils.utils import get_device

torch.multiprocessing.set_sharing_strategy("file_system")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

class Config():
    def __init__(self, batch_size, lr, model, epoch_n):
        self.batch_size = batch_size
        self.lr = lr
        self.model = model
        self.epoch_n = epoch_n

def main():
    """ Run grid search. """

    device = get_device()

    BATCH_SIZES = [16, 32, 64, 128]
    LEARNING_RATES = [0.05, 0.01, 0.005, 0.001]
    MODELS = ["cnn", "mlp"]
    epoch_n = 20

    N = len(BATCH_SIZES)*len(LEARNING_RATES)*len(MODELS)
    i = 1
    config = {}

    for bs in BATCH_SIZES:
        for lr in LEARNING_RATES:
            for model in MODELS:
                
                config = Config(bs, lr, model, epoch_n)

                if config.model == "cnn":
                    model = CNN()
                else:
                    model = MLP()
                
                model.to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
                logging.info(f"Run {i} out of {N}.")
                i += 1

                train_loader, val_loader, _ = create_dataloaders(config.batch_size)
                train_validate(model, config, train_loader, val_loader, optimizer, criterion, device, save = False)


if __name__ == "__main__":
    main()    
