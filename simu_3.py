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
from mpl_toolkits.mplot3d import Axes3D
import pprint

start = time.time()
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


races = db.scalars(
    select(Race)
    .filter(
        Race.kaisai_nen >= "2022",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code >= "00",
        Race.nyusen_tosu > "03",
    )
).all()

wide_graph_2d = []
umaren_graph_2d = []
umatan_graph_2d = []
sanrenpuku_graph_2d = []
sanrentan_graph_2d = []

wide_bet_graph_2d = []
umaren_bet_graph_2d = []
umatan_bet_graph_2d = []
sanrenpuku_bet_graph_2d = []
sanrentan_bet_graph_2d = []

wide_atari_graph_2d = []
umaren_atari_graph_2d = []
umatan_atari_graph_2d = []
sanrenpuku_atari_graph_2d = []
sanrentan_atari_graph_2d = []

wide_benefit_graph_2d = []
umaren_benefit_graph_2d = []
umatan_benefit_graph_2d = []
sanrenpuku_benefit_graph_2d = []
sanrentan_benefit_graph_2d = []


for exp_th in range(10, 14):
    wide_graph = []
    umaren_graph = []
    umatan_graph = []
    sanrenpuku_graph = []
    sanrentan_graph = []

    wide_bet_graph = []
    umaren_bet_graph = []
    umatan_bet_graph = []
    sanrenpuku_bet_graph = []
    sanrentan_bet_graph = []

    wide_atari_graph = []
    umaren_atari_graph = []
    umatan_atari_graph = []
    sanrenpuku_atari_graph = []
    sanrentan_atari_graph = []

    wide_benefit_graph = []
    umaren_benefit_graph = []
    umatan_benefit_graph = []
    sanrenpuku_benefit_graph = []
    sanrentan_benefit_graph = []

    for prob_th in range(18, 24):
        wide_sum = 0
        umaren_sum = 0
        umatan_sum = 0
        sanrenpuku_sum = 0
        sanrentan_sum = 0

        wide_bet = 0
        umaren_bet = 0
        umatan_bet = 0
        sanrenpuku_bet = 0
        sanrentan_bet = 0

        wide_atari = 0
        umaren_atari = 0
        umatan_atari = 0
        sanrenpuku_atari = 0
        sanrentan_atari = 0
        EXP_sum = 0

        for race in races:
            fetcher = Fetcher(
                field_condition=int(race.babajotai_code_dirt)
                + int(race.babajotai_code_shiba),
                kaisai_nen=race.kaisai_nen,
                kaisai_tsukihi=race.kaisai_tsukihi,
                keibajo_code=race.keibajo_code,
                race_bango=race.race_bango,
                db=db,
            )
            fetcher.get_odds_from_db()
            fetcher.setup_odds_df_db()
            odds_df = torch.from_numpy(fetcher.odds_df.to_numpy()).float()
            _y = F.softmax(model(odds_df), dim=1)[:, : fetcher.toroku_tosu]

            y = _y.detach().numpy().copy()[0]
            y = y / y.sum()
            bettor = Bettor(
                rank_prob=y, prob_th=prob_th * 0.1 / int(race.shusso_tosu)
            )
            bettor.setup_probs()
            bettor.set_EXP(fetcher)

            wide_buy = np.where(
                bettor.wide_EXP_low > exp_th * 0.05 + 0.1, 1.0, 0.0
            )
            wide_buy_count = np.sum(wide_buy)
            wide_bet += wide_buy_count
            umaren_buy = np.where(
                bettor.umaren_EXP > exp_th * 0.05 + 0.05, 1.0, 0.0
            )
            umaren_buy_count = np.sum(umaren_buy)
            umaren_bet += umaren_buy_count
            umatan_buy = np.where(
                bettor.umatan_EXP > exp_th * 0.05 + 0.1, 1.0, 0.0
            )
            umatan_buy_count = np.sum(umatan_buy)
            umatan_bet += umatan_buy_count

            sanrenpuku_buy = np.where(
                (bettor.sanrenpuku_EXP > exp_th * 0.05 + 0.05),
                1.0,
                0.0,
            )
            sanrenpuku_buy_count = np.sum(sanrenpuku_buy)
            sanrenpuku_bet += sanrenpuku_buy_count
            sanrentan_buy = np.where(
                (bettor.sanrentan_EXP > exp_th * 0.05),
                1.0,
                0.0,
            )
            sanrentan_buy_count = np.sum(sanrentan_buy)
            sanrentan_bet += sanrentan_buy_count

            back = db.scalars(
                select(BackMoney).filter(
                    BackMoney.kaisai_nen == race.kaisai_nen,
                    BackMoney.kaisai_tsukihi == race.kaisai_tsukihi,
                    BackMoney.keibajo_code == race.keibajo_code,
                    BackMoney.race_bango == race.race_bango,
                )
            ).one()

            wide_result = np.zeros((fetcher.toroku_tosu, fetcher.toroku_tosu))
            wide_back = [
                (
                    int_or_nan(back.haraimodoshi_wide_1a[:2]),
                    int_or_nan(back.haraimodoshi_wide_1a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_2a[:2]),
                    int_or_nan(back.haraimodoshi_wide_2a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_3a[:2]),
                    int_or_nan(back.haraimodoshi_wide_3a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_4a[:2]),
                    int_or_nan(back.haraimodoshi_wide_4a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_5a[:2]),
                    int_or_nan(back.haraimodoshi_wide_5a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_6a[:2]),
                    int_or_nan(back.haraimodoshi_wide_6a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_wide_7a[:2]),
                    int_or_nan(back.haraimodoshi_wide_7a[2:]),
                ),
            ]

            if (not np.isnan(wide_back[0][0])) and wide_back[0][0] != 0:
                wide_result[wide_back[0][0] - 1, wide_back[0][1] - 1] = (
                    float(back.haraimodoshi_wide_1b) * 0.01
                )

            if (not np.isnan(wide_back[1][0])) and wide_back[1][0] != 0:
                wide_result[wide_back[1][0] - 1, wide_back[1][1] - 1] = (
                    float(back.haraimodoshi_wide_2b) * 0.01
                )

            if (not np.isnan(wide_back[2][0])) and wide_back[2][0] != 0:
                wide_result[wide_back[2][0] - 1, wide_back[2][1] - 1] = (
                    float(back.haraimodoshi_wide_3b) * 0.01
                )

            if (not np.isnan(wide_back[3][0])) and wide_back[3][0] != 0:
                wide_result[wide_back[3][0] - 1, wide_back[3][1] - 1] = (
                    float(back.haraimodoshi_wide_4b) * 0.01
                )

            if (not np.isnan(wide_back[4][0])) and wide_back[4][0] != 0:
                wide_result[wide_back[4][0] - 1, wide_back[4][1] - 1] = (
                    float(back.haraimodoshi_wide_5b) * 0.01
                )

            if (not np.isnan(wide_back[5][0])) and wide_back[5][0] != 0:
                wide_result[wide_back[5][0] - 1, wide_back[5][1] - 1] = (
                    float(back.haraimodoshi_wide_6b) * 0.01
                )

            if (not np.isnan(wide_back[6][0])) and wide_back[6][0] != 0:
                wide_result[wide_back[6][0] - 1, wide_back[6][1] - 1] = (
                    float(back.haraimodoshi_wide_7b) * 0.01
                )

            wide_result = wide_result - 1.0
            np.nan_to_num(wide_result, copy=False)
            umaren_result = np.zeros(
                (fetcher.toroku_tosu, fetcher.toroku_tosu)
            )

            umaren_back = [
                (
                    int_or_nan(back.haraimodoshi_umaren_1a[:2]),
                    int_or_nan(back.haraimodoshi_umaren_1a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umaren_2a[:2]),
                    int_or_nan(back.haraimodoshi_umaren_2a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umaren_3a[:2]),
                    int_or_nan(back.haraimodoshi_umaren_3a[2:]),
                ),
            ]
            for atari in umaren_back:
                if (not np.isnan(atari[0])) and atari[0] != 0:
                    umaren_result[atari[0] - 1, atari[1] - 1] = 1.0

            umaren_result *= fetcher.umaren_odds
            umaren_result = umaren_result - 1.0
            np.nan_to_num(umaren_result, copy=False)
            umatan_result = np.zeros(
                (fetcher.toroku_tosu, fetcher.toroku_tosu)
            )

            umatan_back = [
                (
                    int_or_nan(back.haraimodoshi_umatan_1a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_1a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umatan_2a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_2a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umatan_3a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_3a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umatan_4a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_4a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umatan_5a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_5a[2:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_umatan_6a[:2]),
                    int_or_nan(back.haraimodoshi_umatan_6a[2:]),
                ),
            ]
            for atari in umatan_back:
                if (not np.isnan(atari[0])) and atari[0] != 0:
                    umatan_result[atari[0] - 1, atari[1] - 1] = 1.0

            umatan_result *= fetcher.umatan_odds
            umatan_result -= 1.0
            np.nan_to_num(umatan_result, copy=False)

            sanrenpuku_result = np.zeros(
                (fetcher.toroku_tosu, fetcher.toroku_tosu, fetcher.toroku_tosu)
            )

            sanrenpuku_back = [
                (
                    int_or_nan(back.haraimodoshi_sanrenpuku_1a[:2]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_1a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_1a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrenpuku_2a[:2]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_2a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_2a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrenpuku_3a[:2]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_3a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrenpuku_3a[4:]),
                ),
            ]
            for atari in sanrenpuku_back:
                if (not np.isnan(atari[0])) and atari[0] != 0:
                    sanrenpuku_result[
                        atari[0] - 1, atari[1] - 1, atari[2] - 1
                    ] = 1.0

            sanrenpuku_result *= fetcher.sanrenpuku_odds
            sanrenpuku_result -= 1.0
            np.nan_to_num(sanrenpuku_result, copy=False)

            sanrentan_result = np.zeros(
                (fetcher.toroku_tosu, fetcher.toroku_tosu, fetcher.toroku_tosu)
            )

            sanrentan_back = [
                (
                    int_or_nan(back.haraimodoshi_sanrentan_1a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_1a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_1a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrentan_2a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_2a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_2a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrentan_3a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_3a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_3a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrentan_4a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_4a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_4a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrentan_5a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_5a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_5a[4:]),
                ),
                (
                    int_or_nan(back.haraimodoshi_sanrentan_6a[:2]),
                    int_or_nan(back.haraimodoshi_sanrentan_6a[2:4]),
                    int_or_nan(back.haraimodoshi_sanrentan_6a[4:]),
                ),
            ]
            for atari in sanrentan_back:
                if (not np.isnan(atari[0])) and atari[0] != 0:
                    sanrentan_result[
                        atari[0] - 1, atari[1] - 1, atari[2] - 1
                    ] = 1.0
            sanrentan_result *= fetcher.sanrentan_odds
            sanrentan_result -= 1.0
            np.nan_to_num(sanrentan_result, copy=False)

            wide_result *= wide_buy
            umaren_result *= umaren_buy
            umatan_result *= umatan_buy
            sanrenpuku_result *= sanrenpuku_buy
            sanrentan_result *= sanrentan_buy

            wide_benefit = np.sum(wide_result)
            umaren_benefit = np.sum(umaren_result)
            umatan_benefit = np.sum(umatan_result)
            sanrenpuku_benefit = np.sum(sanrenpuku_result)
            sanrentan_benefit = np.sum(sanrentan_result)

            wide_atari += np.count_nonzero(wide_result > 0.0)
            umaren_atari += np.count_nonzero(umaren_result > 0.0)
            umatan_atari += np.count_nonzero(umatan_result > 0.0)
            sanrenpuku_atari += np.count_nonzero(sanrenpuku_result > 0.0)
            sanrentan_atari += np.count_nonzero(sanrentan_result > 0.0)

            wide_sum += wide_benefit
            umaren_sum += umaren_benefit
            umatan_sum += umatan_benefit
            sanrenpuku_sum += sanrenpuku_benefit
            sanrentan_sum += sanrentan_benefit
            all_benefit = (
                +wide_benefit
                + umaren_benefit
                + umatan_benefit
                + sanrenpuku_benefit
                + sanrentan_benefit
            )

            if all_benefit != 0:
                print(
                    f"{race.kaisai_nen}/{race.kaisai_tsukihi}/{race.race_bango}レース/{race.shusso_tosu}頭"
                )

                print(f"wide: {wide_benefit}")
                print(f"umaren: {umaren_benefit}")
                print(f"umatan: {umatan_benefit}")
                print(f"sanrenpuku: {sanrenpuku_benefit}")
                print(f"sanrentan: {sanrentan_benefit}")
                print(f"all: {all_benefit}")

        wide_graph.append((wide_bet + wide_sum) / wide_bet)
        umaren_graph.append((umaren_bet + umaren_sum) / umaren_bet)
        umatan_graph.append((umatan_bet + umatan_sum) / umatan_bet)
        sanrenpuku_graph.append(
            (sanrenpuku_bet + sanrenpuku_sum) / sanrenpuku_bet
        )
        sanrentan_graph.append((sanrentan_bet + sanrentan_sum) / sanrentan_bet)

        wide_benefit_graph.append(wide_sum)
        umaren_benefit_graph.append(umaren_sum)
        umatan_benefit_graph.append(umatan_sum)
        sanrenpuku_benefit_graph.append(sanrenpuku_sum)
        sanrentan_benefit_graph.append(sanrentan_sum)

        wide_bet_graph.append(wide_bet)
        umaren_bet_graph.append(umaren_bet)
        umatan_bet_graph.append(umatan_bet)
        sanrenpuku_bet_graph.append(sanrenpuku_bet)
        sanrentan_bet_graph.append(sanrentan_bet)

        wide_atari_graph.append(wide_atari / wide_bet)
        umaren_atari_graph.append(umaren_atari / umaren_bet)
        umatan_atari_graph.append(umatan_atari / umatan_bet)
        sanrenpuku_atari_graph.append(sanrenpuku_atari / sanrenpuku_bet)
        sanrentan_atari_graph.append(sanrentan_atari / sanrentan_bet)

    wide_graph_2d.append(wide_graph)
    umaren_graph_2d.append(umaren_graph)
    umatan_graph_2d.append(umatan_graph)
    sanrenpuku_graph_2d.append(sanrenpuku_graph)
    sanrentan_graph_2d.append(sanrentan_graph)

    wide_bet_graph_2d.append(wide_bet_graph)
    umaren_bet_graph_2d.append(umaren_bet_graph)
    umatan_bet_graph_2d.append(umatan_bet_graph)
    sanrenpuku_bet_graph_2d.append(sanrenpuku_bet_graph)
    sanrentan_bet_graph_2d.append(sanrentan_bet_graph)

    wide_atari_graph_2d.append(wide_atari_graph)
    umaren_atari_graph_2d.append(umaren_atari_graph)
    umatan_atari_graph_2d.append(umatan_atari_graph)
    sanrenpuku_atari_graph_2d.append(sanrenpuku_atari_graph)
    sanrentan_atari_graph_2d.append(sanrentan_atari_graph)

    wide_benefit_graph_2d.append(wide_benefit_graph)
    umaren_benefit_graph_2d.append(umaren_benefit_graph)
    umatan_benefit_graph_2d.append(umatan_benefit_graph)
    sanrenpuku_benefit_graph_2d.append(sanrenpuku_benefit_graph)
    sanrentan_benefit_graph_2d.append(sanrentan_benefit_graph)

pprint.pprint("wide")
pprint.pprint(wide_graph_2d)
pprint.pprint("umaren")
pprint.pprint(umaren_graph_2d)
pprint.pprint("umatan")
pprint.pprint(umatan_graph_2d)
pprint.pprint("sanrenpuku")
pprint.pprint(sanrenpuku_graph_2d)
pprint.pprint("sanrentan")
pprint.pprint(sanrentan_graph_2d)
pprint.pprint("wide_atari_graph_2d")
pprint.pprint(wide_atari_graph_2d)
pprint.pprint("umaren_atari_graph_2d")
pprint.pprint(umaren_atari_graph_2d)
pprint.pprint("umatan_atari_graph_2d")
pprint.pprint(umatan_atari_graph_2d)
pprint.pprint("sanrenpuku_atari_graph_2d")
pprint.pprint(sanrenpuku_atari_graph_2d)
pprint.pprint("sanrentan_atari_graph_2d")
pprint.pprint(sanrentan_atari_graph_2d)
print("benefit")
pprint.pprint(wide_benefit_graph_2d)
pprint.pprint("umaren")
pprint.pprint(umaren_benefit_graph_2d)
pprint.pprint("umatan")
pprint.pprint(umatan_benefit_graph_2d)
pprint.pprint("sanrenpuku")
pprint.pprint(sanrenpuku_benefit_graph_2d)
pprint.pprint("sanrentan")
pprint.pprint(sanrentan_benefit_graph_2d)


wide_graph_np = np.array(wide_graph_2d)
umaren_graph_np = np.array(umaren_graph_2d)
umatan_graph_np = np.array(umatan_graph_2d)
sanrenpuku_graph_np = np.array(sanrenpuku_graph_2d)
sanrentan_graph_np = np.array(sanrentan_graph_2d)


wide_bet_graph_np = np.array(wide_bet_graph_2d)
umaren_bet_graph_np = np.array(umaren_bet_graph_2d)
umatan_bet_graph_np = np.array(umatan_bet_graph_2d)
sanrenpuku_bet_graph_np = np.array(sanrenpuku_bet_graph_2d)
sanrentan_bet_graph_np = np.array(sanrentan_bet_graph_2d)


wide_atari_graph_np = np.array(wide_atari_graph_2d)
umaren_atari_graph_np = np.array(umaren_atari_graph_2d)
umatan_atari_graph_np = np.array(umatan_atari_graph_2d)
sanrenpuku_atari_graph_np = np.array(sanrenpuku_atari_graph_2d)
sanrentan_atari_graph_np = np.array(sanrentan_atari_graph_2d)

color_list = ["b", "g", "r", "c", "m", "y", "k"]

end = time.time()
print(f"実行時間:{end-start}")
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
prob_ax = np.arange(1.8, 2.6, 0.1)
exp_ax = np.arange(0.5, 0.79, 0.05)
X, Y = np.meshgrid(prob_ax, exp_ax)
ax.plot_wireframe(X, Y, wide_graph_np, label="wide", color=color_list[2])
ax.plot_wireframe(X, Y, umaren_graph_np, label="umaren", color=color_list[3])
ax.plot_wireframe(X, Y, umatan_graph_np, label="umatan", color=color_list[4])
ax.plot_wireframe(
    X, Y, sanrenpuku_graph_np, label="sanrenpuku", color=color_list[5]
)
ax.plot_wireframe(
    X, Y, sanrentan_graph_np, label="sanrentan", color=color_list[6]
)
plt.legend()
plt.show()
