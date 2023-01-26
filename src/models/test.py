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

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.view(images.shape[0], -1)

            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)

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
