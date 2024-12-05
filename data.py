# utils/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


# utils/data.py
class PaddedMNIST(Dataset):
    def __init__(self, sequence_length, train=True, add_noise=True):
        self.sequence_length = sequence_length
        self.mnist = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        self.add_noise = add_noise
        self.input_size = 28 * 28  # MNIST image size

    def __len__(self):
        # Return the number of samples in the MNIST dataset
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        # Flatten MNIST image: (1, 28, 28) -> (784,)
        digit = image.view(-1)

        # Create sequence: digit followed by noise
        sequence = torch.zeros(self.sequence_length, self.input_size)
        sequence[0] = digit  # First timestep is the digit

        if self.add_noise:
            # Add Gaussian noise for remaining timesteps
            noise = torch.randn(self.sequence_length - 1, self.input_size)
            sequence[1:] = noise

        return sequence, label

# Training infrastructure
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        batch_size = sequences.size(0)

        optimizer.zero_grad()

        # Forward pass
        hidden_seq, _ = model(sequences)
        # Use final timestep for classification
        final_hidden = hidden_seq[:, -1, :]

        # Add classification head if needed
        outputs = model.classifier(final_hidden)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / total


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            batch_size = sequences.size(0)

            hidden_seq, _ = model(sequences)
            final_hidden = hidden_seq[:, -1, :]
            outputs = model.classifier(final_hidden)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / total


# Main experiment runner
def run_experiment(model, sequence_length, batch_size, epochs, device):
    # Data preparation
    train_dataset = PaddedMNIST(sequence_length=sequence_length, train=True)
    test_dataset = PaddedMNIST(sequence_length=sequence_length, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }