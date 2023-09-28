import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd


class HorseDataModule(pl.LightningDataModule):
    def __init__(self, train_csv_file, test_csv_file, batch_size: int = 32):
        super().__init__()
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_df = pd.read_csv(self.train_csv_file)
        self.test_df = pd.read_csv(self.test_csv_file)
        all = HorseDataset(self.train_df)                                               
        train_size = int(len(all) * 0.8)
        self.train, self.val = random_split(
            all, [train_size, len(all) - train_size]
        )
        self.test = HorseDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )




class HorseDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame):
        self.feature = torch.tensor(data_frame.iloc[:, :-3].values,dtype=torch.float32)
        self.target = torch.tensor(data_frame["first"].values,dtype=torch.long)-1

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]
