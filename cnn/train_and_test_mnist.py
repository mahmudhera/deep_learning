import os
import pandas as pd

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

    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    
if __name__ == "__main__":
    main()