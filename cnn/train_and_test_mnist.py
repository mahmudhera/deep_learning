import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
logger = CSVLogger("logs", name="my_model")

# use high speed
torch.set_float32_matmul_precision('high')

# Dataset Class
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# CNN LightningModule
class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
        # tracking training loss
        self.train_loss_this_epoch = 0.0

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_loss_this_epoch += loss.item()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc)
        self.log("train_loss", self.train_loss_this_epoch)
        self.train_loss_this_epoch = 0.0

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# DataModule for loading and preparing data
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=64, num_workers=4):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(self.train_file) or not os.path.exists(self.test_file):
            raise FileNotFoundError("MNIST CSV files not found!")

    def setup(self, stage=None):
        train_data = pd.read_csv(self.train_file, header=None)
        test_data = pd.read_csv(self.test_file, header=None)

        X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
        y_train = train_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
        y_test = test_data.iloc[:, 0].values

        # Split training/validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

        self.train_dataset = MNISTDataset(X_train, y_train)
        self.val_dataset = MNISTDataset(X_val, y_val)
        self.test_dataset = MNISTDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=self.num_workers)


# Main
if __name__ == "__main__":
    train_file = 'mnist/mnist_train.csv'
    test_file = 'mnist/mnist_test.csv'

    datamodule = MNISTDataModule(train_file, test_file)
    model = LitCNN()

    trainer = Trainer(
        max_epochs=20,
        accelerator='auto',
        #devices='auto',
        log_every_n_steps=1,
        devices=1,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)
    
    trainer.test(model, datamodule=datamodule)

    # Save model
    torch.save(model.state_dict(), "mnist_lightning_model.pth")