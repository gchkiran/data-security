import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import torch.nn.functional as F

# -----------------------------
# Set random seeds for reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Efficient MNIST CNN Model
# -----------------------------
class EfficientMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Dropout
        self.dropout = nn.Dropout(0.15)

        # Fully connected
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        # Flatten & FC
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -----------------------------
# Data partitioning function
# -----------------------------
def partition_data(dataset, num_clients=10):
    """Randomly partition dataset among `num_clients` clients."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    partition_size = dataset_size // num_clients
    partitions = [
        Subset(dataset, indices[i * partition_size:(i + 1) * partition_size])
        for i in range(num_clients)
    ]
    return partitions

# -----------------------------
# Client-side training
# -----------------------------
def train_client(model, train_loader, optimizer, loss_function, device):
    """Train a client model on its local data."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(train_loader), accuracy

# -----------------------------
# Aggregate client models
# -----------------------------
def aggregate_models(models):
    """Average weights from multiple client models."""
    global_model = EfficientMNIST()
    global_state = global_model.state_dict()

    with torch.no_grad():
        for key, param in global_state.items():
            tensors = [m.state_dict()[key] for m in models]

            if "num_batches_tracked" in key:
                global_state[key] = tensors[0]  # keep first for BatchNorm
            elif param.dtype.is_floating_point:
                global_state[key] = torch.stack(tensors, 0).mean(0)
            else:
                global_state[key] = tensors[0]

        global_model.load_state_dict(global_state)

    return global_model

# -----------------------------
# Model evaluation
# -----------------------------
def test_model(model, test_loader, device):
    """Evaluate a model on test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# -----------------------------
# Federated Learning Process
# -----------------------------
def federated_learning():
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 50
    num_clients = 10
    rounds = 10

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Partition training data among clients
    client_train_data = partition_data(train_dataset, num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Initialize global model and loss function
    global_model = EfficientMNIST().to(device)
    loss_function = nn.CrossEntropyLoss()

    # Initialize best model tracking
    best_model = None
    best_accuracy = 0.0


    # Federated Learning loop
    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}")

        local_models = []
        for client_id in range(num_clients):
            client_model = EfficientMNIST().to(device)
            client_model.load_state_dict(global_model.state_dict())  # Start with global weights

            client_train_loader = DataLoader(
                client_train_data[client_id], batch_size=batch_size, shuffle=True, num_workers=8
            )
            optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

            loss, accuracy = train_client(client_model, client_train_loader, optimizer, loss_function, device)
            print(f"Client {client_id + 1}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
            local_models.append(client_model)

        # Aggregate client models
        global_model = aggregate_models(local_models)

        # Evaluate global model
        global_accuracy = test_model(global_model, test_loader, device)
        print(f"Global Model Test Accuracy: {global_accuracy * 100:.2f}%")
        
        # Track and save the best model based on accuracy
        if global_accuracy > best_accuracy:
            best_accuracy = global_accuracy
            best_model = global_model
            # Save the best model (based on accuracy)
            torch.save(best_model.state_dict(), "Level2_best.pth")
            print("Best model saved as 'Level2_best.pth'.")
    print("Federated Learning finished!")

    return global_model

# -----------------------------
# Run federated learning
# -----------------------------
if __name__ == "__main__":
    federated_learning()
