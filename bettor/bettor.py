import numpy as np
from uma_predict.bettor.fetcher_odds import Fetcher

np.set_printoptions(linewidth=1400)


class Bettor:
    def __init__(self, rank_prob: list, prob_th: float = 0.001) -> None:
        self.rank_prob = rank_prob
        shusso_tosu = len(rank_prob)
        self.shusso_tosu = shusso_tosu
        self.prob_th = prob_th
        self.tansho_prob = None
        self.fukusho_prob = None
        self.wide_prob = None
        self.umaren_prob = None
        self.umatan_prob = None
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
        third_rank_prob = self.prob_third_rank(self.rank_prob)

        third_rank_prob = np.where(
            third_rank_prob < self.prob_th, 0, third_rank_prob
        )

        """
        third_rank_prob[0] = np.where(
            third_rank_prob[0] < 1.4/ self.shusso_tosu,
            0,
            third_rank_prob[0],
        )
        third_rank_prob[1] = np.where(
            third_rank_prob[1] < 0.8  / self.shusso_tosu,
            0,
            third_rank_prob[1],
        )
        third_rank_prob[2] = np.where(
            third_rank_prob[2] < 0.4/ self.shusso_tosu,
            0,
            third_rank_prob[2],
        )
        """
        self.tansho_prob = third_rank_prob[0, :].T[0]
        self.fukusho_prob = third_rank_prob.sum(axis=0).T[0]
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

        for i in range(shusso_tosu):
            for j in range(i + 1, shusso_tosu):
                for k in range(j + 1, shusso_tosu):
                    self.sanrenpuku_prob[i, j, k] = (
                        self.sanrentan_prob[i, j, k]
                        + self.sanrentan_prob[i, k, j]
                        + self.sanrentan_prob[j, i, k]
                        + self.sanrentan_prob[j, k, i]
                        + self.sanrentan_prob[k, i, j]
                        + self.sanrentan_prob[k, j, i]
                    )
        """
        self.sanrenpuku_prob=np.where(
            self.sanrenpuku_prob < self.prob_th, 0, self.sanrenpuku_prob
        )
        self.sanrentan_prob=np.where(
            self.sanrentan_prob < self.prob_th, 0, self.sanrentan_prob
        )
        """
        self.umatan_prob = self.sanrentan_prob.sum(axis=2)
        self.umaren_prob = np.triu(self.umatan_prob + self.umatan_prob.T, k=1)
        self.wide_prob = (
            self.sanrenpuku_prob.sum(axis=1)
            + self.sanrenpuku_prob.sum(axis=2)
            + self.sanrenpuku_prob.sum(axis=0)
        )

    def set_EXP(self, fetcher: Fetcher):
        self.tansho_EXP = (
            self.tansho_prob * fetcher.tansho_odds + self.tansho_prob - 1
        )
        self.fukusho_EXP_low = (
            self.fukusho_prob * fetcher.fukusho_odds_low
            + self.fukusho_prob
            - 1
        )

        self.fukusho_EXP_up = self.fukusho_prob * fetcher.fukusho_odds_up
        self.wide_EXP_low = self.wide_prob * fetcher.wide_odds_low

        self.wide_EXP_up = self.wide_prob * fetcher.wide_odds_up
        self.umaren_EXP = self.umaren_prob * fetcher.umaren_odds

        self.umatan_EXP = self.umatan_prob * fetcher.umatan_odds

        self.sanrenpuku_EXP = (
            self.sanrenpuku_prob * fetcher.sanrenpuku_odds
            + self.sanrenpuku_prob
            - 1
        )
        self.sanrentan_EXP = (
            self.sanrentan_prob * fetcher.sanrentan_odds
            + self.sanrentan_prob
            - 1
        )

    def setup_probs_v2(self):
        self.tansho_prob = self.rank_prob
        shusso_tosu = len(self.rank_prob)
        rank_prob = np.where(
            self.rank_prob < self.prob_th, 0, self.rank_prob
        )
        for i in range(shusso_tosu):
            for j in range(shusso_tosu):
                for k in range(shusso_tosu):
                    if i != j and j != k and k != i:
                        self.sanrentan_prob[i, j, k] = (
                            rank_prob[i]
                            * (rank_prob[j] / (1 - rank_prob[i]))
                            * (
                                rank_prob[k]
                                / (1 - rank_prob[i] - rank_prob[j])
                            )
                        )

        for i in range(shusso_tosu):
            for j in range(i + 1, shusso_tosu):
                for k in range(j + 1, shusso_tosu):
                    self.sanrenpuku_prob[i, j, k] = (
                        self.sanrentan_prob[i, j, k]
                        + self.sanrentan_prob[i, k, j]
                        + self.sanrentan_prob[j, i, k]
                        + self.sanrentan_prob[j, k, i]
                        + self.sanrentan_prob[k, i, j]
                        + self.sanrentan_prob[k, j, i]
                    )
        """
        self.sanrenpuku_prob=np.where(
            self.sanrenpuku_prob < self.prob_th, 0, self.sanrenpuku_prob
        )
        self.sanrentan_prob=np.where(
            self.sanrentan_prob < self.prob_th, 0, self.sanrentan_prob
        )
        """
        self.umatan_prob = self.sanrentan_prob.sum(axis=2)
        self.umaren_prob = np.triu(self.umatan_prob + self.umatan_prob.T, k=1)
        self.wide_prob = (
            self.sanrenpuku_prob.sum(axis=1)
            + self.sanrenpuku_prob.sum(axis=2)
            + self.sanrenpuku_prob.sum(axis=0)
        ) / 3
        self.fukusho_prob = (
            (self.wide_prob + self.wide_prob.T).sum(axis=1)
        ) / 2

    def update_probs(self, new_rank_prob: list):
        self.rank_prob = new_rank_prob
        self.setup_probs()

    def disp_EXP(self):
        print(f"tansho: {self.tansho_EXP}")
        print(f"fukusho_low: {self.fukusho_EXP_low}")
        print(f"fukusho_up: {self.fukusho_EXP_up}")
        print(f"wide_low: {self.wide_EXP_low}")
        print(f"wide_up: {self.wide_EXP_up}")
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
        fukusho_high = np.where(self.fukusho_EXP_low > threshold)
        for i in fukusho_high[0]:
            print(f"fukusho:{i+1}: {self.fukusho_EXP_low[i]}")
        wide_high = np.where(self.wide_EXP_low > threshold)
        for i, j in zip(wide_high[0], wide_high[1]):
            print(f"wide:{i+1}-{j+1}: {self.wide_EXP_low[i,j]}")
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
            "fukusho": np.where(self.fukusho_EXP_low > threshold),
            "wide": np.where(self.wide_EXP_low > threshold),
            "umaren": np.where(self.umaren_EXP > threshold),
            "umatan": np.where(self.umatan_EXP > threshold),
            "sanrenpuku": np.where(self.sanrenpuku_EXP > threshold),
            "sanrentan": np.where(self.sanrentan_EXP > threshold),
        }

    def prob_third_rank(self, rank_prob: np.ndarray):
        sort_rank_prob = np.sort(rank_prob)[::-1]
        rank = np.argsort(-rank_prob)
        prob_mat = np.zeros((len(rank_prob), len(rank_prob)))
        prob_mat[0, :] = sort_rank_prob
        prob_mat[1:, 0] = sort_rank_prob[1:]
        for i in range(1, 3):
            prob_mat[i, i] = 1 - prob_mat[i, :i].sum()
            for j in range(i, len(rank_prob) - 1):
                if prob_mat[j:, 0].sum() == 0:
                    break

                ratio = prob_mat[j, 0] / (prob_mat[j:, 0].sum())
                prob_mat[j + 1, i] = prob_mat[j, i] * (1 - ratio)
                prob_mat[j, i] = prob_mat[j, i] * ratio
            prob_mat[-1, i] = 1 - prob_mat[:-1, i].sum()
            prob_mat[i, i + 1 :] = prob_mat[i + 1 :, i]
        restore_rank = [np.where(rank == i)[0] for i in range(len(rank_prob))]
        prob_mat = prob_mat[:, restore_rank]

        return prob_mat[:3, :]

    def prob_third_rank_v2(self, rank_prob: np.ndarray):
        sort_rank_prob = np.sort(rank_prob)[::-1]
        rank = np.argsort(-rank_prob)
        prob_mat = np.zeros((len(rank_prob), len(rank_prob)))
        chakusa_mat = sort_rank_prob.copy() / sort_rank_prob[0]
        prob_mat[0, :] = chakusa_mat
        prob_mat[1:, 0] = chakusa_mat[1:]
        prob_mat[1, 1:] = chakusa_mat[1:] / chakusa_mat[1]
        prob_mat[2:, 1] = prob_mat[1, 2:]
        prob_mat[2, 2:] = prob_mat[1, 2:] / prob_mat[1, 2]
        prob_mat = prob_mat / prob_mat.sum(axis=1).reshape(-1, 1)
        restore_rank = [np.where(rank == i)[0] for i in range(len(rank_prob))]
        print(prob_mat[:3, :])
        prob_mat = prob_mat[:, restore_rank]

        return prob_mat[:3, :]
