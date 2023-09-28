import torch
from data3 import HorseDataModule
from predictor3  import HorsePredictor
from pytorch_lightning import Trainer
import pandas as pd

def main():
    data=pd.read_csv("df/data5/2012-2021_5.csv")
    column_len=data.shape[1]-3
    torch.manual_seed(2)
    horse_data = HorseDataModule("df/data5/2012-2021_5.csv", "df/data5/2022_5.csv")
    horse_data.setup()
    horse_predictor = HorsePredictor(column_len, 18)

    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, max_epochs=200)
    else:
        trainer=Trainer(max_epochs=200)

    trainer.fit(horse_predictor, horse_data)
    trainer.test(horse_predictor, horse_data)

if __name__ == '__main__':
    main()