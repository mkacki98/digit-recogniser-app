import torch.nn as nn
import torch
import numpy as np
import torch.multiprocessing

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.create_dataset import create_dataloaders
from src.models.model import MLP
from src.utils import load_configs, display_training_examples

torch.multiprocessing.set_sharing_strategy("file_system")


def main():

    config = load_configs()

    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader, val_loader, test_loader = create_dataloaders(config.batch_size)

    display_training_examples()
    train_validate(model, config, train_loader, val_loader, optimizer, criterion)

    torch.save(
        model,
        f"models/mnist_classifier_{config.batch_size}_{config.epoch_n}",
    )


def train_validate(model, config, train_loader, val_loader, optimizer, criterion):
    """Train the model given config parameters, optimizer, loss function and dataloaders."""

    run_name = f"lr-{config.lr}_batch_size-{config.batch_size}_epoch_n-{config.epoch_n}"
    writer = SummaryWriter(f"logs/fit/{run_name}")

    best_val_loss = np.inf

    for epoch in tqdm(range(config.epoch_n)):

        training_loss = epoch_train(
            model, train_loader, optimizer, criterion, writer, epoch
        )
        validation_loss = epoch_validate(model, val_loader, criterion)

        epoch_avg_train_loss = round(training_loss / len(train_loader), 5)
        epoch_avg_val_loss = round(validation_loss / len(val_loader), 5)

        print(
            f"EPOCH {epoch+1} TRAIN LOSS | {epoch_avg_train_loss} | VALIDATION LOSS | {epoch_avg_val_loss} |"
        )

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

    for i, data in tqdm(enumerate(train_loader), position=0, leave=True):

        images, labels = data
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        running_loss += loss.item()

        _, predicted = torch.max(predictions.data, 1)
        running_hits += (predicted == labels).sum().item()

        if (i + 1) % 1000 == 0:

            print(f" \n Epoch {i+1}, step {i+1}, loss {loss.item()}.")
            running_accuracy = round(running_hits / 1000 / predicted.size(0), 4)

            writer.add_scalar(
                "Training Loss (steps)", running_loss / 1000, epoch * n_steps + i
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
    model.eval()

    for i, data in enumerate(val_loader):
        image, label = data
        image = image.view(image.shape[0], -1)

        prediction = model(image)

        loss = criterion(prediction, label)
        val_loss += loss.item()

    return val_loss


if __name__ == "__main__":
    main()
