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


train_data = pl.read_parquet("df\dataOdds\\2016-2021-30_v3.parquet")
column_len = train_data.shape[1] - 18

model = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_123/checkpoints/epoch=119-step=29160.ckpt",
    input_size=column_len,
    output_size=18,
    hiden_layer_size=[1024, 512, 256, 32],
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
fetcher.setup_odds_df_db()
odds_df = torch.from_numpy(fetcher.odds_df.to_numpy()).float()

_y = F.softmax(model(odds_df), dim=1)[:, : fetcher.toroku_tosu]
_y = _y.detach().numpy().copy()[0]
y = _y / _y.sum()
for i, j in enumerate(y):
    print(f"{i+1}番: {j}")
bettor = Bettor(rank_prob=y, prob_th=1.95 / int(fetcher.shusso_tosu))
bettor.setup_probs()
bettor.set_EXP(fetcher)

tansho_buy = np.where(bettor.tansho_EXP > 0.5, 1.0, 0.0)
fukusho_buy = np.where(bettor.fukusho_EXP_low > 0.5, 1.0, 0.0)
wide_buy = np.where(bettor.wide_EXP_low > 0.65, 1.0, 0.0)
umaren_buy = np.where(bettor.umaren_EXP > 0.7, 1.0, 0.0)
umatan_buy = np.where(bettor.umatan_EXP > 0.7, 1.0, 0.0)
sanrenpuku_buy = np.where(
    (bettor.sanrenpuku_EXP > 0.6),
    1.0,
    0.0,
)
sanrentan_buy = np.where(
    (bettor.sanrentan_EXP > 0.6),
    1.0,
    0.0,
)

print(f"{fetcher.race_name_abbre}/{fetcher.race_bango}レース")
wide_ticket = np.where(wide_buy == 1.0)
for i, j in zip(wide_ticket[0], wide_ticket[1]):
    print(f"ワイド{i+1}-{j+1}")
    print(f"{bettor.wide_EXP_low[i,j]}")
umaren_ticket = np.where(umaren_buy == 1.0)
for i, j in zip(umaren_ticket[0], umaren_ticket[1]):
    print(f"馬連{i+1}-{j+1}")
    print(f"{bettor.umaren_EXP[i,j]}")

sanrentan_ticket = np.where(sanrentan_buy == 1.0)
for i, j, k in zip(
    sanrentan_ticket[0], sanrentan_ticket[1], sanrentan_ticket[2]
):
    print(f"三連単{i+1}-{j+1}-{k+1}")
    print(f"{bettor.sanrentan_EXP[i,j,k]}")
