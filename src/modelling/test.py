import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from utils.utils import load_configs, get_model_name, get_device

def main():
    config = load_configs()
    test(config)

def test(config):
    """Test the model on unseen data."""

    correct = 0
    total = 0
    
    device = get_device()
    model = torch.load(f"models/{get_model_name(config)}")
    test_loader = torch.load("data/test_loader.pkl")
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):            
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)
            _, predicted = torch.max(prediction.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * round(correct/total, 4)} %.")


if __name__ == "__main__":
    main()
