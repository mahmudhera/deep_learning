import os
import pandas as pd
import numpy as np

import torch

training_filename = 'mnist/mnist_train.csv'
testing_filename = 'mnist/mnist_test.csv'


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
    
    # convert the data to pytorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    
    # show the first example in the training set, and the test set, and the validation set
    print("First training example label:", y_train[0], "with shape:", X_train[0].shape)
    print("First validation example label:", y_val[0], "with shape:", X_val[0].shape)
    print("First testing example label:", y_test[0], "with shape:", X_test[0].shape)
    
    
    
if __name__ == "__main__":
    main()