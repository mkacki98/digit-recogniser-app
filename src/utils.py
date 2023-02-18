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

    parser.add_argument(
        "--mlp",
        help="Architecture of the network.",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--cnn",
        help="Architecture of the network (CNN or MLP).",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    return args

def get_model_name(config):
    """ Get the name of the model to be saved. """

    model_name = ""
    if config.cnn:
        model_name += "2cl-1fc_"
    else:
        model_name += "2fc_"

    model_name += "bs-" + str(config.batch_size) + "_"
    model_name += "lr-" + str(config.lr) + "_"
    model_name += "epoch-" + str(config.epoch_n)

    return model_name



