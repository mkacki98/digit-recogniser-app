import torch

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


def create_dataloaders(batch_size):
    """Download the data if it's not at `root` yet and create dataloaders."""

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_val_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    test_data, val_data = random_split(test_val_data, [3000, 7000])

    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    torch.save(test_loader, "data/test_loader.pkl")

    return train_loader, val_loader, test_loader
