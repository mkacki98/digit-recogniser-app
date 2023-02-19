import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch.nn as nn
import torch
import torch.multiprocessing
import logging

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.dataset.create_dataset import create_dataloaders
from src.modelling.models import MLP, CNN
from src.utils import load_configs, display_training_examples, get_model_name

torch.multiprocessing.set_sharing_strategy("file_system")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def train_validate(model, config, train_loader, val_loader, optimizer, criterion, device, save = True):
    """Train the model given config parameters, optimizer, loss function and dataloaders."""

    logging.info(f"\n Started training a {config.model.upper()} for {config.epoch_n} epochs with batch size of {config.batch_size}, learning rate of {config.lr}.")

    writer = SummaryWriter(f"logs/fit/{config.model} | BatchSize{config.batch_size} | LearningRate {config.lr} | Epochs {config.epoch_n}")

    with trange(config.epoch_n) as t:
        for epoch in t:

            epoch_train_loss, epoch_train_acc = epoch_train(
                model, train_loader, optimizer, criterion, writer, epoch, device
            )

            epoch_val_loss, epoch_val_acc = epoch_validate(model, val_loader, criterion, writer, epoch, device)

            t.set_postfix(train_acc=epoch_train_acc, val_acc = epoch_val_acc)
            
            writer.add_scalar(
                "epoch train loss", epoch_train_loss, epoch
            )
            writer.add_scalar(
                "epoch val loss", epoch_val_loss, epoch
            )
            writer.flush()

    if save:
        torch.save(
            model,
            f"models/{get_model_name(config)}",
        )

def epoch_train(model, train_loader, optimizer, criterion, writer, epoch, device):
    """Train the model through a single data pass (epoch). Report the result after every `reports_n` examples."""

    model.train(True)

    total = 0
    running_hits = 0
    running_loss = 0.0
    n_steps = len(train_loader)
    accuracies = []
    losses = []

    model.eval()

    for i, data in enumerate(train_loader):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction = model(images)

        loss = criterion(prediction, labels)     

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        running_hits += (predicted == labels).sum().item()

        report = 50

        if (i + 1) % report == 0:

            acc = (running_hits / total) # acc derived from the past 10 batches
            loss = (running_loss / total)  # avg loss per example in the past 10 batches
            
            accuracies.append(acc)
            losses.append(loss)
            
            writer.add_scalar(
                "train loss", loss, epoch * n_steps + i
            )
            writer.add_scalar(
                "train accuracy", acc, epoch * n_steps + i
            )

            running_loss = 0.0
            running_hits = 0
            total = 0

    # Report average loss and accuracy per epoch
    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accuracies)/len(accuracies)
    
    model.train(False)

    return epoch_loss, epoch_acc

def epoch_validate(model, val_loader, criterion, writer, epoch, device):
    """Collect the validation loss for that epoch."""

    total = 0

    running_hits = 0
    running_loss = 0.0

    n_steps = len(val_loader)
    model.eval()

    accuracies = []
    losses = []

    for i, data in enumerate(val_loader):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)

        loss = criterion(prediction, labels)
        running_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        running_hits += (predicted == labels).sum().item()

        report = 50

        if (i + 1) % report == 0:
            
            acc = (running_hits / total) # acc derived from the past 10 batches
            loss = (running_loss / total)  # avg loss per example in the past 10 batches
            
            accuracies.append(acc)
            losses.append(loss)
            
            writer.add_scalar(
                "val loss", loss, epoch * n_steps + i
            )
            writer.add_scalar(
                "val accuracy", acc, epoch * n_steps + i
            )

            running_loss = 0.0
            running_hits = 0
            total = 0

    # Report average loss and accuracy per epoch
    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accuracies)/len(accuracies)
    
    return epoch_loss, epoch_acc

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    config = load_configs()

    if config.model == "cnn":
        model = CNN()
    else:
        model = MLP()
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader, val_loader, _ = create_dataloaders(config.batch_size)

    display_training_examples()
    train_validate(model, config, train_loader, val_loader, optimizer, criterion, device)

if __name__ == "__main__":

    main()
