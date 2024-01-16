import torch
from data5 import HorseDataModule
from predictor5 import HorsePredictor
from pytorch_lightning import Trainer
import polars as pl


def main():
    train_data = pl.read_parquet("df\dataOdds\\2016-2021-30_v3.parquet")
    test_data = pl.read_parquet("df\dataOdds\\2022-1_v2.parquet")
    print(train_data.shape)
    print(test_data.shape)
    column_len = train_data.shape[1] - 18
    print(column_len)
    torch.manual_seed(1)
    horse_data = HorseDataModule(
        train_data=train_data, test_data=test_data, batch_size=2048
    )
    horse_data.setup()
    horse_predictor = HorsePredictor(
        column_len, 18, hiden_layer_size=[1024, 512, 256,32]
    )

    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, max_epochs=200)
    else:
        trainer = Trainer(max_epochs=120)

    trainer.fit(horse_predictor, horse_data)
    trainer.test(horse_predictor, horse_data)


if __name__ == "__main__":
    main()
