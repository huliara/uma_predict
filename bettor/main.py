from uma_predict.network.predictor4 import HorsePredictor
import numpy as np
import pandas as pd
from fetcher import Fetcher
from uma_predict.bettor.bettor import Bettor
import os
import datetime
import polars as pl
import torch.nn.functional as F
target_race_key = 202106030811
shusso_tosu = 18

train_data=pl.read_parquet("df\data10\\2016-2021-5-10.parquet")
column_len=train_data.shape[1]-18

model = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_76/checkpoints/epoch=49-step=7150.ckpt",
    input_size=column_len,
    output_size=18,
)
model.eval()
fetcher = Fetcher(
    kaisai_nen="2022",
    kaisai_tsukihi="1225",
    keibajo_code="06",
    race_bango="11",
)
fetcher.get_horse_data_from_db()
fetcher.get_odds_from_db()
y = F.softmax(model(fetcher.horse_data[:,:-18])[:,:fetcher.toroku_tosu]).detach().numpy().copy()[0]
bettor = Bettor(rank_prob=y)
bettor.setup_probs()
bettor.set_EXP(fetcher)

bettor.disp_high_EXP(threshold=10.)

"""
dir_path = f"{str(target_race_key)})/{datetime.datetime.now().strftime('%Y-%m-%d;%H:%M')}"

os.makedirs(
    dir_path,
    exist_ok=True,
)

pd.DataFrame(EV["tansho"]).to_csv(dir_path + "tansho.csv")
pd.DataFrame(EV["fukusho"]).to_csv(dir_path + "fukusho.csv")
pd.DataFrame(EV["wide"]).to_csv(dir_path + "wide.csv")
pd.DataFrame(EV["umaren"]).to_csv(dir_path + "umaren.csv")
pd.DataFrame(EV["umatan"]).to_csv(dir_path + "umatan.csv")
for i in range(0, shusso_tosu):
    pd.DataFrame(EV["sanrenpuku"][i]).to_csv(
        dir_path + "/sanrenpuku/" + f"{i+1}.csv"
    )
    pd.DataFrame(EV["sanrentan"][i]).to_csv(
        dir_path + "/sanrentan/" + f"{i+1}.csv"
    )
"""
