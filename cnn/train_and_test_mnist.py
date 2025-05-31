import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


training_filename = 'mnist/mnist_train.csv'
testing_filename = 'mnist/mnist_test.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    """
    Load the MNIST dataset from CSV files.
    """
    if not os.path.exists(training_filename) or not os.path.exists(testing_filename):
        raise FileNotFoundError("MNIST dataset files not found. Please ensure they are in the correct directory.")

    train_data = pd.read_csv(training_filename, header=None)
    test_data = pd.read_csv(testing_filename, header=None)
    
    # there are no headers

    X_train = train_data.iloc[:, 1:].values  # All columns except the first one
    y_train = train_data.iloc[:, 0].values  # The first column
    X_test = test_data.iloc[:, 1:].values  # All columns except the first one
    y_test = test_data.iloc[:, 0].values  # The first column
    
    # the data are 784 pixels, reshape them to 28x28
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize to [0, 1]
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize to [0, 1]
    
    # partition train data into training and validation sets
    validation_data_fraction = 0.1
    validation_size = int(len(X_train) * validation_data_fraction)
    
    # randomly select validation data
    indices = np.random.permutation(len(X_train))
    X_train, X_val = X_train[indices[:-validation_size]], X_train[indices[-validation_size:]]
    y_train, y_val = y_train[indices[:-validation_size]], y_train[indices[-validation_size:]]

    return X_train, y_train, X_val, y_val, X_test, y_test



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # (B, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(2, 2)               # (B, 32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # (B, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, 2)               # (B, 64, 7, 7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    



def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    
    # show the first example in the training set, and the test set, and the validation set
    print("First training example label:", y_train[0], "with shape:", X_train[0].shape)
    print("First validation example label:", y_val[0], "with shape:", X_val[0].shape)
    print("First testing example label:", y_test[0], "with shape:", X_test[0].shape)
    
    # create datasets
    train_dataset = MNISTDataset(X_train.numpy(), y_train.numpy())
    val_dataset = MNISTDataset(X_val.numpy(), y_val.numpy())
    test_dataset = MNISTDataset(X_test.numpy(), y_test.numpy())
    
    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # initialize model, loss function, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # show which device is being used
    print(f"Using device: {device}")
    
    # main training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    
    
    
if __name__ == "__main__":
    main()