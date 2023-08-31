import torch
from data import HorseDataModule
from predictor import HorsePredictor
from pytorch_lightning import Trainer
import pandas as pd

def main():
    data=pd.read_csv("df/data2/2006_2021.csv")
    column_len=data.shape[1]-54
    torch.manual_seed(1)
    horse_data = HorseDataModule("df/data2/2006_2021.csv", "df/data2/2022.csv")
    horse_data.setup()
    horse_predictor = HorsePredictor(column_len, 54)

    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, max_epochs=100)
    else:
        trainer = Trainer(max_epochs=100)

    trainer.fit(horse_predictor, horse_data)

if __name__ == '__main__':
    main()