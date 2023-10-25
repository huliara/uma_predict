from uma_predict.network.predictor4 import HorsePredictor
import numpy as np
import pandas as pd
from fetcher import Fetcher
from uma_predict.bettor.bettor import Bettor
import os
import datetime

target_race_key = 202106030811
shusso_tosu = 18


def expected_value(prob_mat: np.array, odds: np.array):
    return (prob_mat * odds).sum()


model = HorsePredictor.load_from_checkpoint(
    "uma_predict/lightning_logs/version_74/checkpoints/epoch=29-step=4290.ckpt"
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
y = model(fetcher.horse_data).detach().numpy().copy()
bettor = Bettor(rank_prob=y)
bettor.setup_probs()
bettor.set_EXP(fetcher)

bettor.disp_EXP()
bettor.disp_high_EXP()

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
