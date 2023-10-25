import numpy as np
from fetcher import Fetcher


class Bettor:
    def __init__(self, rank_prob: list) -> None:
        self.rank_prob = rank_prob
        shusso_tosu = len(rank_prob)
        self.tansho_prob = None
        self.fukusho_prob = None
        self.wide_prob = np.zeros((shusso_tosu, shusso_tosu))
        self.umaren_prob = np.zeros((shusso_tosu, shusso_tosu))
        self.umatan_prob = np.zeros((shusso_tosu, shusso_tosu))
        self.sanrenpuku_prob = np.zeros(
            (shusso_tosu, shusso_tosu, shusso_tosu)
        )
        self.sanrentan_prob = np.zeros((shusso_tosu, shusso_tosu, shusso_tosu))
        self.tansho_EXP = None
        self.fukusho_EXP_low = None
        self.fukusho_EXP_up = None
        self.wide_EXP_low = None
        self.wide_EXP_up = None
        self.umaren_EXP = None
        self.umatan_EXP = None
        self.sanrenpuku_EXP = None
        self.sanrentan_EXP = None

    def setup_probs(self):
        third_rank_prob = Bettor.prob_third_rank(self.rank_prob)
        self.tansho_prob = third_rank_prob[0, :]
        self.fukusho_prob = third_rank_prob.sum(axis=0)
        shusso_tosu = len(self.rank_prob)
        for i in range(shusso_tosu):
            for j in range(shusso_tosu):
                for k in range(shusso_tosu):
                    if i != j and j != k and k != i:
                        self.sanrentan_prob[i, j, k] = (
                            third_rank_prob[0, i]
                            * (
                                third_rank_prob[1, j]
                                / (1 - third_rank_prob[0, j])
                            )
                            * (
                                third_rank_prob[2, k]
                                / (1 - third_rank_prob[:2, k].sum())
                            )
                        )

        pre_sanrenpuku = (
            self.sanrentan_prob
            + self.sanrentan_prob.transpose(0, 2, 1)
            + self.sanrentan_prob.transpose(1, 0, 2)
            + self.sanrentan_prob.transpose(1, 2, 0)
            + self.sanrentan_prob.transpose(2, 0, 1)
            + self.sanrentan_prob.transpose(2, 1, 0)
        )

        pre_sanrenpuku = np.triu(pre_sanrenpuku, k=1)
        pre_sanrenpuku = np.rot90(pre_sanrenpuku, k=1, axes=(1, 2))
        pre_sanrenpuku = np.triu(pre_sanrenpuku, k=1)
        pre_sanrenpuku = np.rot90(pre_sanrenpuku, k=1, axes=(1, 0))
        pre_sanrenpuku = np.triu(pre_sanrenpuku, k=1)
        self.sanrenpuku_prob = np.rot90(
            np.rot90(pre_sanrenpuku, k=1, axes=(0, 1)), k=1, axes=(2, 1)
        )

        self.umatan_prob = self.sanrentan_prob.sum(axis=2)
        self.umaren_prob = np.triu(self.umatan_prob + self.umatan_prob.T, k=1)
        self.wide_prob = self.sanrenpuku_prob.sum(axis=0)

    def set_EXP(self, fetcher: Fetcher):
        self.tansho_EXP = self.tansho_prob * fetcher.tansho_odds
        self.fukusho_EXP_low = self.fukusho_prob * fetcher.fukusho_odds_low
        self.fukusho_EXP_up = self.fukusho_prob * fetcher.fukusho_odds_up
        self.wide_EXP_low = self.wide_prob * fetcher.wide_odds_low
        self.wide_EXP_up = self.wide_prob * fetcher.wide_odds_up
        self.umaren_EXP = self.umaren_prob * fetcher.umaren_odds
        self.umatan_EXP = self.umatan_prob * fetcher.umatan_odds
        self.sanrenpuku_EXP = self.sanrenpuku_prob * fetcher.sanrenpuku_odds
        self.sanrentan_EXP = self.sanrentan_prob * fetcher.sanrentan_odds

    def update_probs(self, new_rank_prob: list):
        self.rank_prob = new_rank_prob
        self.setup_probs()

    def disp_EXP(self):
        print(f"tansho: {self.tansho_EXP}")
        print(f"fukusho: {self.fukusho_EXP}")
        print(f"wide: {self.wide_EXP}")
        print(f"umaren: {self.umaren_EXP}")
        print(f"umatan: {self.umatan_EXP}")
        print(f"sanrenpuku: {self.sanrenpuku_EXP}")
        print(f"sanrentan: {self.sanrentan_EXP}")

    def disp_prob(self):
        print(f"tansho: {self.tansho_prob}")
        print(f"fukusho: {self.fukusho_prob}")
        print(f"wide: {self.wide_prob}")
        print(f"umaren: {self.umaren_prob}")
        print(f"umatan: {self.umatan_prob}")
        print(f"sanrenpuku: {self.sanrenpuku_prob}")
        print(f"sanrentan: {self.sanrentan_prob}")

    def disp_high_EXP(self, threshold: float = 1.0):
        tansho_high = np.where(self.tansho_EXP > threshold)
        for i in tansho_high[0]:
            print(f"tansho:{i+1}: {self.tansho_EXP[i]}")
        fukusho_high = np.where(self.fukusho_EXP > threshold)
        for i in fukusho_high[0]:
            print(f"fukusho:{i+1}: {self.fukusho_EXP[i]}")
        wide_high = np.where(self.wide_EXP > threshold)
        for i, j in zip(wide_high[0], wide_high[1]):
            print(f"wide:{i+1}-{j+1}: {self.wide_EXP[i,j]}")
        umaren_high = np.where(self.umaren_EXP > threshold)
        for i, j in zip(umaren_high[0], umaren_high[1]):
            print(f"umaren:{i+1}-{j+1}: {self.umaren_EXP[i,j]}")
        umatan_high = np.where(self.umatan_EXP > threshold)
        for i, j in zip(umatan_high[0], umatan_high[1]):
            print(f"umatan:{i+1}→{j+1}: {self.umatan_EXP[i,j]}")
        sanrenpuku_high = np.where(self.sanrenpuku_EXP > threshold)
        for i, j, k in zip(
            sanrenpuku_high[0], sanrenpuku_high[1], sanrenpuku_high[2]
        ):
            print(
                f"sanrenpuku:{i+1}-{j+1}-{k+1}: {self.sanrenpuku_EXP[i,j,k]}"
            )
        sanrentan_high = np.where(self.sanrentan_EXP > threshold)
        for i, j, k in zip(
            sanrentan_high[0], sanrentan_high[1], sanrentan_high[2]
        ):
            print(f"sanrentan:{i+1}→{j+1}→{k+1}: {self.sanrentan_EXP[i,j,k]}")

    def high_exp_tickets(self, threshold: float = 1.0):
        return {
            "tansho": np.where(self.tansho_EXP > threshold),
            "fukusho": np.where(self.fukusho_EXP > threshold),
            "wide": np.where(self.wide_EXP > threshold),
            "umaren": np.where(self.umaren_EXP > threshold),
            "umatan": np.where(self.umatan_EXP > threshold),
            "sanrenpuku": np.where(self.sanrenpuku_EXP > threshold),
            "sanrentan": np.where(self.sanrentan_EXP > threshold),
        }

    @staticmethod
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
