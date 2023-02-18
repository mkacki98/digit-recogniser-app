import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import argparse

def main():
    config = load_config()
    test(config)


def test(config):
    """Test the model on unseen data."""

    correct = 0
    total = 0

    model = torch.load(f"models/{config.model_name}")
    test_loader = torch.load("data/test_loader.pkl")
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):            
            images, labels = data

            prediction = model(images)
            _, predicted = torch.max(prediction.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * round(correct/total, 4)} %.")


def load_config():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        help="Model name to be tested.",
        type=str,
        default="mnist_classifier_base",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
