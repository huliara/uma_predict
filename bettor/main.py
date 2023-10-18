from uma_predict.network.predictor4 import HorsePredictor
import numpy as np
from fetcher import Fetcher


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
fetcher=Fetcher()
target_race_data=fetcher.get_raceinfo(202106030811)
target_odds=fetcher.get_odds(202106030811)
y=model(target_race_data)
third_rank_prob = prob_third_rank(y)

EV=expected_value(third_rank_prob,target_odds)
