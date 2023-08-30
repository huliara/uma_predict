import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

class HorsePredictor(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_hat, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(y_hat, y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)