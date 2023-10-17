import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from typing import Optional
import torch.nn.functional as F


class HorsePredictor(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        hiden_layer_size: list[int] = [2048,512,32],
    ):
        super().__init__()
        self.train_acc = Accuracy(task="multiclass", num_classes=18)
        self.val_acc = Accuracy(task="multiclass", num_classes=18)
        self.test_acc = Accuracy(task="multiclass", num_classes=18)
        all_layers = []
        first_layersize = input_size
        for unit in hiden_layer_size:
            layer = nn.Linear(first_layersize, unit)
            all_layers.append(layer)
            all_layers.append(nn.LeakyReLU())
            first_layersize = unit
        all_layers.append(nn.Linear(hiden_layer_size[-1], output_size))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred_1 = torch.argmax(y_hat, dim=1)
        first = torch.argmax(y, dim=1)
        self.train_acc.update(pred_1, first)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_training_epoch_end(self, outputs):
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred_1 = torch.argmax(y_hat, dim=1)
        first = torch.argmax(y, dim=1)
        self.val_acc.update(pred_1, first)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        pred_1 = torch.argmax(y_hat, dim=1)
        first = torch.argmax(y, dim=1)
        self.test_acc.update(pred_1, first)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001,weight_decay=0.0004)
