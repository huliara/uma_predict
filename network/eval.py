from uma_predict.network.predictor4 import HorsePredictor
from 
import numpy as np
import pandas as pd
import os
import datetime

target_race_key = 202106030811
shusso_tosu = 18


def prob_third_rank(rank_prob: list):
    sort_rank_prob = np.sort(rank_prob)
    rank = np.argsort(rank_prob)
    prob_mat = np.zeros((18, 18))
    prob_mat[0, :] = sort_rank_prob
    prob_mat[1:, 0] = sort_rank_prob[1:]
    for i in range(1, 3):
        prob_mat[i, i] = 1 - prob_mat[i, :i].sum()
        for j in range(i + 1, 18):
            ratio = prob_mat[j, 0] / (prob_mat[j:, 0].sum())
            prob_mat[j, i] = prob_mat[j - 1, i] * (1 - ratio)
            prob_mat[j - 1, i] = prob_mat[j - 1, i] * ratio
        prob_mat[i, i + 1 :] = prob_mat[i + 1 :, i]
    restore_rank = [np.where(rank == i)[0] for i in range(18)]
    return prob_mat[:3, restore_rank]


def expected_value(prob_mat: np.array, odds: np.array):
    return (prob_mat * odds).sum()


model = HorsePredictor.load_from_checkpoint(
    "uma_predict/lightning_logs/version_74/checkpoints/epoch=29-step=4290.ckpt"
)
model.eval()
fetcher = Fetcher()
target_race_data = fetcher.get_raceinfo(202106030811)
target_odds = fetcher.get_odds(202106030811)
y = model(target_race_data)
third_rank_prob = prob_third_rank(y)

EV = expected_value(rank_prob, target_odds)

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