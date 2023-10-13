import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import polars.dataframe as dd


class HorseDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data, batch_size: int = 32):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        all = HorseDataset(self.train_data)                                  
        train_size = int(len(all) * 0.8)
        self.train, self.val = random_split(
            all, [train_size, len(all) - train_size]
        )
        self.test = HorseDataset(self.test_data)

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
        self.feature = torch.tensor(data_frame.get_columns()[:-18],dtype=torch.float32).T
        self.target = torch.tensor(data_frame.get_columns()[-18:],dtype=torch.float32).T
    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]
