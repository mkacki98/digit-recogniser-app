import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch.nn as nn
import torch
import numpy as np
import torch.multiprocessing

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.dataset.create_dataset import create_dataloaders
from src.modelling.models import MLP, CNN
from src.utils import load_configs, display_training_examples, get_model_name

torch.multiprocessing.set_sharing_strategy("file_system")

def train_validate(model, config, train_loader, val_loader, optimizer, criterion):
    """Train the model given config parameters, optimizer, loss function and dataloaders."""

    run_name = f"lr-{config.lr}_batch-{config.batch_size}_epochs-{config.epoch_n}"
    writer = SummaryWriter(f"logs/fit/{run_name}")

    best_val_loss = np.inf

    with trange(config.epoch_n) as t:
        for epoch in t:

            t.set_description(f"batch_size-{config.batch_size}-lr-{config.lr}")

            training_loss = epoch_train(
                model, train_loader, optimizer, criterion, writer, epoch
            )
            validation_loss, validation_acc = epoch_validate(model, val_loader, criterion)

            epoch_avg_train_loss = round(training_loss / len(train_loader), 5)
            epoch_avg_val_loss = round(validation_loss / len(val_loader), 5)
            epoch_avg_acc = round(validation_acc, 5)

            t.set_postfix(train_loss = epoch_avg_train_loss, val_loss = epoch_avg_val_loss, val_acc = epoch_avg_acc)

            writer.add_scalars(
                "avg_loss",
                {
                    "train": epoch_avg_train_loss,
                    "valid": epoch_avg_val_loss,
                },
                epoch + 1,
            )
            writer.flush()

            if epoch_avg_train_loss < best_val_loss:
                best_val_loss = epoch_avg_train_loss
                model_path = f"models/model"
                torch.save(model.state_dict(), model_path)

def epoch_train(model, train_loader, optimizer, criterion, writer, epoch):
    """Train the model through a single data pass (epoch). Report the result after every `reports_n` examples."""

    model.train(True)

    train_loss = 0.0
    n_steps = len(train_loader)

    running_loss = 0.0
    running_hits = 0

    for i, data in enumerate(train_loader):

        images, labels = data

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        running_loss += loss.item()

        _, predicted = torch.max(predictions.data, 1)
        running_hits += (predicted == labels).sum().item()

        report = 100
        if (i + 1) % report == 0:

            running_accuracy = round(running_hits / report / predicted.size(0), 4)
            train_loss = running_loss / report

            writer.add_scalar(
                "Training Loss (steps)", train_loss, epoch * n_steps + i
            )
            writer.add_scalar(
                "Training Accuracy (steps)", running_accuracy, epoch * n_steps + i
            )

            running_loss = 0.0
            running_hits = 0

    model.train(False)

    return train_loss

def epoch_validate(model, val_loader, criterion):
    """Collect the validation loss for that epoch."""

    val_loss = 0.0
    total = 0
    correct = 0
    model.eval()

    for i, data in enumerate(val_loader):
        images, labels = data

        prediction = model(images)

        loss = criterion(prediction, labels)
        val_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return val_loss, correct/total

def main():

    config = load_configs()

    if config.mlp:
        model = MLP()
    if config.cnn:
        model = CNN()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader, val_loader, _ = create_dataloaders(config.batch_size)

    display_training_examples()
    train_validate(model, config, train_loader, val_loader, optimizer, criterion)
    
    torch.save(
        model,
        f"models/{get_model_name(config)}",
    )

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    main()
