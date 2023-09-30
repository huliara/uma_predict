import torch
from data3 import HorseDataModule
from predictor3  import HorsePredictor
from pytorch_lightning import Trainer
import pandas as pd

def main():
    data=pd.read_csv("df/data4/2012_2021_2.csv",index_col=0)
    column_len=data.shape[1]-3
    torch.manual_seed(1)
    horse_data = HorseDataModule("df/data4/2012_2021_2.csv", "df/data4/2012_2021_2.csv")
    horse_data.setup()
    horse_predictor = HorsePredictor(column_len, 18)

    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, max_epochs=200)
    else:
        trainer=Trainer(max_epochs=50)

    trainer.fit(horse_predictor, horse_data)
    trainer.test(horse_predictor, horse_data)

if __name__ == '__main__':
    main()