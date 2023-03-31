import argparse
import torchvision
import base64
import cv2
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

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
        help="Number of epochs while training.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--model",
        help="Architecture of the model - `mlp`, `cnn` or `nmf_clf`.",
        type=str,
        default="mlp",
    )

    parser.add_argument(
        "--hid",
        help="Size of the hidden layer in a neuromorphic model.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--hid_epoch_n", help="Number of epochs while training synapses.", type=int, default=10
    )

    parser.add_argument(
        "--delta",
        help="Stength of the inhibition (Anti-Hebbian learning).",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--p",
        help="Lebesgue norm, used for activation when training synapses.",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--rank",
        help="Defines how many hidden units the model considers in a competition.",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--tau_l",
        help="Defines the time scale of the process.",
        type=float,
        default=1e-30,
    )

    args = parser.parse_args()

    return args

def get_model_name(config):
    """Get the name of the model to be saved. """

    model_name = ""
    if config.model == "cnn":
        model_name += "2cl-1fc_"
    elif config.model == "mlp":
        model_name += "2fc_"
    else:
        model_name += "nmf_1fc_"
        model_name += "hid-" + str(config.hid) + "_"

    model_name += "bs-" + str(config.batch_size) + "_"
    model_name += "lr-" + str(round(config.lr,3)) + "_"
    model_name += "epoch-" + str(config.epoch_n)

    return model_name

def get_image(canvas):
    """Decode the canvas and pass it as OpenCV image. """

    decoded_canvas = base64.b64decode(canvas.split(',')[1].encode())
    canvas_as_np = np.frombuffer(decoded_canvas, dtype=np.uint8)
    
    img = cv2.imdecode(canvas_as_np, flags=1)
    resized_img = cv2.resize(img,(28,28))
    
    return resized_img

def predict_image(image):
    """Load a model and use it to predict the image. """

    device = get_device()

    model = torch.load("models/mnist_classifier_base")
    model.eval()

    image = image[:,:,0] # (28,28)
    image = (torch.from_numpy(image)/1.0)  
    image = torch.unsqueeze(torch.unsqueeze(image, 0), 0) # (1,1,28,28)

    predictions = model(normalize(image).to(device))

    return predictions, torch.argmax(predictions)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
    
def nmf_activation_fn(p, synapses):
    """Apply activation function on synapses, f() from (1). """
    
    sign = torch.sign(synapses)
    return sign * torch.absolute(synapses) ** (p-1)
