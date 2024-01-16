import numpy as np


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
    syusso_tosu = len(prob_mat)
    sanrentan_prob=np.zeros((syusso_tosu,syusso_tosu,syusso_tosu))
    third_rank=prob_third_rank(prob_mat)
    


    result={
        "tansho":[],
        "fukusho":[],
        "wide",
        "umaren":[],
        "umatan":[],
        "sanrenpuku":[],
        "sanrentan":[]
    }