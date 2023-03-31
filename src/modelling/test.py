import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from src.utils.utils import load_configs, get_model_name, get_device

def test(config):
    """Test the model on unseen data."""

    correct = 0
    total = 0
    
    device = get_device()
    model = torch.load(f"models/{get_model_name(config)}")
    is_neuromorphic = model.__class__.__name__ == "NeuromorphicClassifier"

    test_loader = torch.load("data/test_loader.pkl")
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):            
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if is_neuromorphic:
                images = images.flatten(start_dim=1, end_dim = 3)

            prediction = model(images)

            if is_neuromorphic:
                predicted = torch.argmax(prediction, 1)
            else:
                _, predicted = torch.max(prediction.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * round(correct/total, 4)} %.")

def main():
    config = load_configs()
    test(config)

if __name__ == "__main__":
    main()
