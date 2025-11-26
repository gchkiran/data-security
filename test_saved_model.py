import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model_definition import EfficientMNIST 
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size_test=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=2
    )

    print(f"Testing has {len(test_loader)} batches")
    return test_loader


# -----------------------------
# Load saved model
# -----------------------------
def load_trained_model(model_path="Level1.pth"):
    model = EfficientMNIST().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def test_saved_model():
    model = load_trained_model("Level1.pth")
    test_loader = load_data()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nFinal Test Accuracy: {correct / total:.8f}")
if __name__ == "__main__":
    test_saved_model()