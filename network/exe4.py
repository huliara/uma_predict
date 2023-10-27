import torch
from data4 import HorseDataModule
from predictor4  import HorsePredictor
from pytorch_lightning import Trainer
import polars as pl

def main():
    train_data=pl.read_parquet("df\data10\\2016-2021-5-10.parquet")
    test_data=pl.read_parquet("df\data10\\2022-5-1.parquet")
    print(train_data.shape)
    print(test_data.shape)
    column_len=train_data.shape[1]-18
    print(column_len)
    torch.manual_seed(1)
    horse_data = HorseDataModule(train_data=train_data, test_data=test_data,batch_size=1024)
    horse_data.setup()
    horse_predictor = HorsePredictor(column_len, 18,hiden_layer_size=[2048,512,32])

    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, max_epochs=200)
    else:
        trainer=Trainer(max_epochs=62)

    trainer.fit(horse_predictor, horse_data)
    trainer.test(horse_predictor, horse_data)

if __name__ == '__main__':
    main()