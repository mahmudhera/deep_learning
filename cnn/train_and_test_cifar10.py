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
import pickle

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
logger = CSVLogger("cifar10_logs", name="cnn_model")

# use high speed
torch.set_float32_matmul_precision('high')

# Dataset Class
class Cifar10Dataset(Dataset):
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
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16

            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 → 8x8

            nn.AdaptiveAvgPool2d((1, 1))  # output: (batch, 64, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        
        # tracking training loss
        self.train_loss_this_epoch = 0.0

    def forward(self, x):
        # TODO
        x = self.net(x)
        x = self.classifier(x)
        return x

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
        # compute validation loss
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss)
        self.log("train_loss", self.train_loss_this_epoch)
        self.train_loss_this_epoch = 0.0
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    
# DataModule for loading and preparing data
class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, train_file_list, test_file, batch_size=256, num_workers=4):
        super().__init__()
        self.train_file_list = train_file_list
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(self.test_file):
            raise FileNotFoundError("files not found!")
        for train_file in self.train_file_list:
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Training file {train_file} not found!")

    def setup(self, stage=None):
        X_train = None
        y_train = None
        for train_file in self.train_file_list:
            with open(train_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                dict = {k.decode('utf-8'): v for k, v in dict.items()}
                X_train = np.concatenate((X_train, dict['data']), axis=0) if X_train is not None else dict['data']
                y_train = np.concatenate((y_train, dict['labels']), axis=0) if y_train is not None else dict['labels']
            
        # show one X_train
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        with open(self.test_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            dict = {k.decode('utf-8'): v for k, v in dict.items()}
            X_test = dict['data']
            y_test = dict['labels']
            print(X_test[0])
            
        # reshape X_train and X_test
        X_train = X_train.reshape(-1, 32, 32, 3).astype(np.float32) / 255.0
        X_test = X_test.reshape(-1, 32, 32, 3).astype(np.float32) / 255.0

        # Split training/validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

        self.train_dataset = Cifar10Dataset(X_train, y_train)
        self.val_dataset = Cifar10Dataset(X_val, y_val)
        self.test_dataset = Cifar10Dataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=self.num_workers)
    
    
# Main
if __name__ == "__main__":
    train_files = ['cifar10/cifar-10-batches-py/data_batch_1',
                   'cifar10/cifar-10-batches-py/data_batch_2',
                   'cifar10/cifar-10-batches-py/data_batch_3',
                   'cifar10/cifar-10-batches-py/data_batch_4',
                   'cifar10/cifar-10-batches-py/data_batch_5']
    test_file = 'cifar10/cifar-10-batches-py/test_batch'

    datamodule = Cifar10DataModule(train_files, test_file)
    model = LitCNN()

    trainer = Trainer(
        max_epochs=50,
        accelerator='auto',
        #devices='auto',
        log_every_n_steps=1,
        devices=1,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # Save model
    #torch.save(model.state_dict(), "mnist_lightning_model.pth")