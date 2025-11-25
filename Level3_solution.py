import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import random
import numpy as np

# -----------------------------
# Set random seed for reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Efficient MNIST Model
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


def partition_data(dataset, num_clients=10):
    """Randomly partition dataset into `num_clients` subsets."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    partition_size = dataset_size // num_clients
    partitions = [
        Subset(dataset, indices[i * partition_size:(i + 1) * partition_size])
        for i in range(num_clients)
    ]
    return partitions


def train_client(model, train_loader, optimizer, loss_function, device):
    """Train a client model on its local data."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

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


def test_model(model, test_loader, device):
    """Evaluate the global model on the test dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total



# -----------------------------
# Aggregation with Distance-Based Filter
# -----------------------------
def aggregate_with_distance_filter(global_model, client_updates, round_num, malicious_client_id=None, malicious_start_round=5, detection_factor=1.5):
    """Aggregate updates, detect malicious clients, and update the global model."""
    # Simulate malicious client if required
    if round_num >= malicious_start_round and malicious_client_id is not None:
        print(f"Malicious client {malicious_client_id + 1} introduced.")
        malicious_update = {
            key: torch.randn_like(value).float() if value.dtype.is_floating_point else value
            for key, value in client_updates[malicious_client_id].items()
        }
        client_updates[malicious_client_id] = malicious_update

    # Compute average update
    global_keys = global_model.state_dict().keys()
    avg_model_updates = {
        key: torch.mean(torch.stack([update[key].float() for update in client_updates]), dim=0)
        for key in global_keys
    }

    # Compute distances of each client update from average
    distances = [
        sum(torch.norm(update[key].float() - avg_model_updates[key]).item() for key in global_keys)
        for update in client_updates
    ]

    # Detection threshold
    threshold = detection_factor * sum(distances) / len(distances)
    print(f"Malicious detection threshold: {threshold:.2f}")

    # Filter valid updates
    valid_updates = [update for i, update in enumerate(client_updates) if distances[i] <= threshold]
    detected_malicious = [i for i, d in enumerate(distances) if d > threshold]

    for mc in detected_malicious:
        print(f"Malicious client detected: {mc + 1}")

    # Aggregate valid updates
    aggregated_updates = {
        key: torch.mean(torch.stack([update[key].float() for update in valid_updates]), dim=0)
        for key in global_keys
    }

    global_model.load_state_dict(aggregated_updates)
    return global_model, valid_updates, detected_malicious


# -----------------------------
# Federated Learning with Malicious client detection
# -----------------------------
def federated_learning_with_MCD():
    learning_rate = 0.001
    batch_size = 50
    num_clients = 10
    rounds = 10

    # Data transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Partition training data
    client_train_data = partition_data(train_dataset, num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize global model and loss
    global_model = EfficientMNIST().to(device)
    loss_function = nn.CrossEntropyLoss()

    # Select a random malicious client
    malicious_client_id = random.randint(0, num_clients - 1)

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1}/{rounds} ---")
        client_updates = []

        # Train each client locally
        for client_id in range(num_clients):
            client_model = EfficientMNIST().to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_train_loader = DataLoader(client_train_data[client_id], batch_size=batch_size, shuffle=True)
            optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

            # Train client
            loss, acc = train_client(client_model, client_train_loader, optimizer, loss_function, device)
            print(f"Client {client_id + 1}: Loss={loss:.4f}, Accuracy={acc*100:.2f}%")
            client_updates.append(client_model.state_dict())

        # Aggregate updates with distance-based filtering
        global_model, _, detected_clients = aggregate_with_distance_filter(global_model, client_updates, round_num, malicious_client_id)

        # Evaluate global model
        global_acc = test_model(global_model, test_loader, device)
        print(f"Global Model Test Accuracy: {global_acc*100:.2f}%")

    # Save final global model
    torch.save(global_model.state_dict(), "Level3.pth")
    print("\nGlobal model saved as 'Level3.pth'.")


# -----------------------------
# Run Federated Learning
# -----------------------------
if __name__ == "__main__":
    federated_learning_with_MCD()
