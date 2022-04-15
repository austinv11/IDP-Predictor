import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utilities import move_off_cpu, plot_loss_accuracy
from idp_dataset import IdpDataset, DatasetMode, AAs


class BasicMLP(nn.Module):

    def __init__(self):
        super(BasicMLP, self).__init__()

        self.embedding = nn.Embedding(len(AAs), 26)
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding on just the first dimension
        letter_embedding = self.embedding(x[:, 0])
        # Concatenate the letter embedding with the rest of the input
        x = torch.cat((letter_embedding, x[:, 1:]), dim=1)
        # Pass through the first hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        # Pass through the second hidden layer
        x = self.fc2(x)
        x = self.relu(x)
        # Pass through the output layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Convenience function to calculate total testing loss and accuracy
def calc_test_accuracy(model, test_loader, loss_fn):
    # Trackers
    test_accuracy = 0
    test_loss = 0
    # Test loop
    with torch.no_grad():
        for X, y in test_loader:
            # Move to GPU
            X, y = move_off_cpu(X), move_off_cpu(y)

            # Forward pass
            y_pred = model(X).squeeze()
            # Compute loss
            loss = loss_fn(y_pred, y.float())

            # Compute test loss and accuracy
            test_loss += loss.item()
            test_accuracy += (y_pred.round() == y).sum().item()

    # Compute test loss and accuracy over all the batches
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    batch_size = 64

    train_dataset = IdpDataset(only_binary_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = IdpDataset(DatasetMode.VALIDATION, only_binary_labels=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    # Move to GPU if available
    model = move_off_cpu(BasicMLP())

    # Binary cross entropy loss
    loss_fn = nn.BCELoss()
    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    # Track loss and accuracy per epoch
    epoch2loss = []
    epoch2acc = []
    testepoch2loss = []
    testepoch2acc = []

    # Train the model
    print("### TRAINING ###")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        train_acc = 0
        # Batches
        for batch, (X, y) in enumerate(train_loader):
            # Move to GPU
            X, y = move_off_cpu(X), move_off_cpu(y)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X).squeeze()
            # Compute loss
            loss = loss_fn(y_pred, y.float())
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute training loss and accuracy
            train_loss += loss.item()
            train_acc += (y_pred.round() == y).sum().item()

            # Print progress every 10 batches
            if batch % 100 == 0:
                print(f"\tBatch {batch+1}/{len(train_loader)}: "
                      f"Loss {loss.item():.4f}")

        # Compute training loss and accuracy per epoch
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        epoch2loss.append(train_loss)
        epoch2acc.append(train_acc)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_acc:.4f}")

        # Compute testing loss and accuracy per epoch
        # To track progress
        test_loss, test_acc = calc_test_accuracy(model,
                                                 validation_loader,
                                                 loss_fn)
        testepoch2loss.append(test_loss)
        testepoch2acc.append(test_acc)

    plot_loss_accuracy([epoch2loss, testepoch2loss],
                       [epoch2acc, testepoch2acc],
                       names=["Training", "Validation"], plot_name="Basic MLP")


if __name__ == "__main__":
    main()
