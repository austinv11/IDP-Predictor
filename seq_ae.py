from random import randint

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from idp_dataset import IdpDataset, DatasetMode, AAs, INDEX_TO_AA
from utilities import move_off_cpu, sliding_window, plot_loss_accuracy


class AminoAcidAE(nn.Module):

    def __init__(self,
                 sequence_length: int = 128,
                 latent_dim: int = 128):
        super(AminoAcidAE, self).__init__()

        self.input_dim = sequence_length*len(AAs)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            self.weight_init(nn.Linear(self.input_dim, self.input_dim*2, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim*2),
            self.weight_init(nn.Linear(self.input_dim*2, latent_dim*4, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim*4),
            self.weight_init(nn.Linear(latent_dim*4, latent_dim*2, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim*2),

            self.weight_init(nn.Linear(latent_dim*2, latent_dim)),
        )

        self.decoder = nn.Sequential(
            self.weight_init(nn.Linear(latent_dim, latent_dim*2, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim*2),
            self.weight_init(nn.Linear(latent_dim*2, latent_dim*4, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim*4),
            self.weight_init(nn.Linear(latent_dim*4, self.input_dim*2, bias=False)),
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim*2),

            self.weight_init(nn.Linear(self.input_dim*2, self.input_dim)),
            nn.Unflatten(1, (sequence_length, len(AAs))),
            nn.Sigmoid()
        )

    def weight_init(self, module=None):
        # Linear init
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', a=0.0003)
            if module.bias is not None:
                nn.init.normal_(module.bias.data)
        return module

    def sequence2embedding(self, x):
        x = move_off_cpu(torch.tensor([AAs[a] for a in x]))
        z = self.encoder(aas2tensor(x).float().unsqueeze(0))
        return z

    def embedding2sequence(self, z):
        x = self.decoder(z)
        x = tensor2aas(x)
        return "".join([INDEX_TO_AA[a.item()] for a in x.squeeze(0).int()])

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def aas2tensor(aas: torch.Tensor) -> torch.Tensor:
    return F.one_hot(aas, num_classes=len(AAs)).float()


def tensor2aas(tensor: torch.Tensor) -> torch.Tensor:
    return torch.argmax(tensor, dim=2)


def remove_prefix_gaps(aas: torch.Tensor) -> torch.Tensor:
    # Find first non-zero index
    first_non_zero = torch.argmax((torch.sum(aas, dim=2) > 1e-20).int(), dim=1)
    for i in range(first_non_zero.size(0)):
        if first_non_zero[i] > 0:
            aas[i, :, :] = torch.roll(aas[i, :, :], -first_non_zero[i], dims=1)
    return aas


# Convenience function to calculate total testing loss and accuracy
def calc_test_accuracy(model, test_loader, loss_fn, window_size):
    # Trackers
    test_accuracy = 0
    test_loss = 0
    total_size = 0
    # Test loop
    with torch.no_grad():
        model.eval()
        for X, y in test_loader:
            # Move to GPU
            X = move_off_cpu(X)

            X = sliding_window(X, window_size, flatten=False, centered=False).squeeze(0)
            X = aas2tensor(X)

            # Forward pass
            Xpred = model(X).squeeze()
            # Compute loss
            X = remove_prefix_gaps(X)
            loss = loss_fn(X, Xpred)

            # Compute test loss and accuracy
            test_loss += loss.item()
            test_accuracy += torch.sum(torch.flatten(tensor2aas(X)).int().eq(torch.flatten(tensor2aas(Xpred)).int())).item()
            total_size += X.shape[0]*X.shape[1]
    model.train()

    # Compute test loss and accuracy over all the batches
    test_loss /= len(test_loader)
    test_accuracy /= total_size
    return test_loss, test_accuracy


def train_ae(window_size=127,
             embedding_size=128,
             epochs=20,
             learning_rate=1e-4,
             weight_decay=1e-3) -> AminoAcidAE:

    train_dataset = IdpDataset(only_binary_labels=True, only_sequences=True)
    test_dataset = IdpDataset(DatasetMode.VALIDATION, only_binary_labels=True, only_sequences=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ae = AminoAcidAE(window_size, embedding_size)
    ae = move_off_cpu(ae)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Track loss and accuracy per epoch
    epoch2loss = []
    epoch2acc = []
    testepoch2loss = []
    testepoch2acc = []

    test_sequence = "".join([INDEX_TO_AA[randint(1, 9)] for i in range(window_size)])

    # Train the model
    print("### TRAINING ###")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        train_acc = 0
        total_size = 0
        # Batches
        for batch, (X, y) in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()

            # Move to GPU
            X = move_off_cpu(X)
            X = sliding_window(X, window_size, flatten=False, centered=False)

            # Shuffle the windows
            shuffled_indices = torch.randperm(X.size(1))
            X = X[:, shuffled_indices].squeeze(0)
            X = aas2tensor(X)

            # Forward pass
            Xpred = ae(X)
            # Compute loss, Trim padding from beginning and place it at the end
            X = remove_prefix_gaps(X)
            loss = loss_fn(X, Xpred)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute training loss and accuracy
            train_loss += loss.item()
            train_acc += torch.sum(torch.flatten(tensor2aas(X)).int().eq(torch.flatten(tensor2aas(Xpred)).int())).item()
            total_size += X.shape[0]*X.shape[1]

            # Print progress every 10 batches
            if batch % 100 == 0:
                print(f"\tBatch {batch + 1}/{len(train_loader)}: "
                      f"Loss {loss.item():.4f}")

        # Compute training loss and accuracy per epoch
        train_loss /= len(train_loader)
        train_acc /= total_size
        epoch2loss.append(train_loss)
        epoch2acc.append(train_acc)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"\nGiven sequence: {test_sequence}")
        ae.eval()
        with torch.no_grad():
            seq_embed = ae.sequence2embedding(test_sequence)
            pred_sequence = ae.embedding2sequence(seq_embed)
        ae.train()
        print(f"Embedding: {seq_embed}")
        print(f"Predicted sequence: {pred_sequence}\n")

        # Compute testing loss and accuracy per epoch
        # To track progress
        test_loss, test_acc = calc_test_accuracy(ae,
                                                 test_loader,
                                                 loss_fn,
                                                 window_size)
        testepoch2loss.append(test_loss)
        testepoch2acc.append(test_acc)

    plot_loss_accuracy([epoch2loss, testepoch2loss],
                       [epoch2acc, testepoch2acc],
                       names=["Training", "Validation"], plot_name="Auto Encoder")

    return ae


if __name__ == "__main__":
    train_ae()
