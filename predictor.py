from uma_predict.bettor.fetcher import Fetcher
import numpy as np
import torch.nn.functional as F
from uma_predict.network.predictor4 import HorsePredictor
import polars as pl
from uma_predict.bettor.bettor import Bettor
from uma_predict.db.database import SessionLocal
from uma_predict.db.models import Race, Career, BackMoney
from sqlalchemy.future import select
import torch
import matplotlib.pyplot as plt
import time

db = SessionLocal()
np.set_printoptions(linewidth=np.inf)


train_data = pl.read_parquet("df\dataC\\2016-2021-5-10.parquet")
column_len = train_data.shape[1] - 18

model = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_83_cbest/checkpoints/epoch=28-step=4147.ckpt",
    input_size=column_len,
    output_size=18,
)
model.eval()


def int_or_nan(num: int):
    try:
        return int(num)
    except:
        return np.nan


fetcher = Fetcher(
    field_condition=1,
    kaisai_nen="2023",
    kaisai_tsukihi="1105",
    race_bango="12",
    race_name_abbre="3回京都2日",
)
fetcher.get_recent_odds_from_jra()
fetcher.get_horse_data_from_jra()
_y = F.softmax(model(fetcher.horse_data), dim=1)[
    :, : fetcher.toroku_tosu
]
_y = _y.detach().numpy().copy()[0]
y = _y / _y.sum()
for i, j in enumerate(y):
    print(f"{i+1}番: {j}")
bettor = Bettor(rank_prob=y, prob_th=1.8 / int(fetcher.shusso_tosu))
bettor.setup_probs()
bettor.set_EXP(fetcher)

tansho_buy = np.where(bettor.tansho_EXP > 1.2, 1.0, 0.0)
fukusho_buy = np.where(bettor.fukusho_EXP_low > 1.2, 1.0, 0.0)
wide_buy = np.where(bettor.wide_EXP_low > 1.2, 1.0, 0.0)
umaren_buy = np.where(bettor.umaren_EXP > 1.2, 1.0, 0.0)
umatan_buy = np.where(bettor.umatan_EXP > 1.2, 1.0, 0.0)
sanrenpuku_buy = np.where(
    (bettor.sanrenpuku_EXP > 1.4),
    1.0,
    0.0,
)
sanrentan_buy = np.where(
    (bettor.sanrentan_EXP > 1.4),
    1.0,
    0.0,
)
bettor.disp_high_EXP(threshold=1.4)

print(f"{fetcher.race_name_abbre}/{fetcher.race_bango}レース")
sanrenpuku_ticket = np.where(sanrenpuku_buy == 1.0)
for i, j, k in zip(
    sanrenpuku_ticket[0], sanrenpuku_ticket[1], sanrenpuku_ticket[2]
):
    print(f"三連複{i+1}-{j+1}-{k+1}")
    print(f"{bettor.sanrenpuku_EXP[i,j,k]}")
sanrentan_ticket = np.where(sanrentan_buy == 1.0)
for i, j, k in zip(
    sanrentan_ticket[0], sanrentan_ticket[1], sanrentan_ticket[2]
):
    print(f"三連単{i+1}-{j+1}-{k+1}")
    print(f"{bettor.sanrentan_EXP[i,j,k]}")