import argparse
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader


def display_training_examples():
    """Display training examples in Tensorboard."""

    writer = SummaryWriter("logs/data/mnist")

    data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    data_loader = DataLoader(data, batch_size=256, shuffle=True, num_workers=0)

    example = iter(data_loader)
    example_data, _ = next(example)

    mnist = torchvision.utils.make_grid(example_data, nrow=16)
    writer.add_image("mnist_images", mnist)
    writer.close()


def load_configs():
    """Load model parameters from the command line."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", help="Size of a training batch.", type=int, default=16
    )
    parser.add_argument(
        "--lr", help="Learning rate of the optimiser.", type=float, default=0.001
    )
    parser.add_argument(
        "--epoch_n",
        help="Number of epochs (full passes through the train data) while training.",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    return args
