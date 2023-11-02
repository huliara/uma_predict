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

start = time.time()
db = SessionLocal()
np.set_printoptions(linewidth=np.inf)
prob_th = 0.15

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


races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen >= "2016",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code <= "26",
        Race.track_code >= "00",
        Race.nyusen_tosu > "03",
        Race.kyoso_joken_code != "701",
    )
).all()

tansho_sum = 0
fukusho_sum = 0
wide_sum = 0
umaren_sum = 0
umatan_sum = 0
sanrenpuku_sum = 0
sanrentan_sum = 0

tansho_bet = 0
fukusho_bet = 0
wide_bet = 0
umaren_bet = 0
umatan_bet = 0
sanrenpuku_bet = 0
sanrentan_bet = 0

tansho_atari = 0
fukusho_atari = 0
wide_atari = 0
umaren_atari = 0
umatan_atari = 0
sanrenpuku_atari = 0
sanrentan_atari = 0


tansho_graph = []
fukusho_graph = []
wide_graph = []
umaren_graph = []
umatan_graph = []
sanrenpuku_graph = []
sanrentan_graph = []
race_sum = [0] * 14
race_bet = [0] * 14
shusso_tosu_sum = [0] * 18
shusso_tosu_bet = [0] * 18
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
    fetcher.get_horse_data_from_db()
    _y = F.softmax(model(fetcher.horse_data[:, :-18]), dim=1)[
        :, : fetcher.toroku_tosu
    ]

    y = _y.detach().numpy().copy()[0]
    y = y / y.sum()
    bettor = Bettor(rank_prob=y,prob_th=1.8/ int(race.shusso_tosu))
    bettor.setup_probs()
    bettor.set_EXP(fetcher)

    tansho_buy = np.where(bettor.tansho_EXP > 1.4, 1.0, 0.0)
    tansho_buy_count = np.sum(tansho_buy)
    tansho_bet += tansho_buy_count
    fukusho_buy = np.where(bettor.fukusho_EXP_low > 1.4, 1.0, 0.0)
    fukusho_buy_count = np.sum(fukusho_buy)
    fukusho_bet += fukusho_buy_count
    wide_buy = np.where(bettor.wide_EXP_low > 1.4, 1.0, 0.0)
    wide_buy_count = np.sum(wide_buy)
    wide_bet += wide_buy_count
    umaren_buy = np.where(bettor.umaren_EXP > 1.4, 1.0, 0.0)
    umaren_buy_count = np.sum(umaren_buy)
    umaren_bet += umaren_buy_count
    umatan_buy = np.where(bettor.umatan_EXP > 1.4, 1.0, 0.0)
    umatan_buy_count = np.sum(umatan_buy)
    umatan_bet += umatan_buy_count

    sanrenpuku_buy = np.where(
        (bettor.sanrenpuku_EXP > 1.4),
        1.0,
        0.0,
    )
    sanrenpuku_buy_count = np.sum(sanrenpuku_buy)
    sanrenpuku_bet += sanrenpuku_buy_count
    sanrentan_buy = np.where(
        (bettor.sanrentan_EXP > 1.4),
        1.0,
        0.0,
    )
    sanrentan_buy_count = np.sum(sanrentan_buy)
    sanrentan_bet += sanrentan_buy_count

    shusso_tosu_bet[int(race.shusso_tosu) - 1] += (
        +sanrenpuku_buy_count + sanrentan_buy_count
    )
    race_bet[int(race.race_bango) - 1] += (
        sanrenpuku_buy_count + sanrentan_buy_count
    )
    back = db.scalars(
        select(BackMoney).filter(
            BackMoney.kaisai_nen == race.kaisai_nen,
            BackMoney.kaisai_tsukihi == race.kaisai_tsukihi,
            BackMoney.keibajo_code == race.keibajo_code,
            BackMoney.race_bango == race.race_bango,
        )
    ).one()

    tansho_back = [
        int_or_nan(back.haraimodoshi_tansho_1a),
        int_or_nan(back.haraimodoshi_tansho_2a),
        int_or_nan(back.haraimodoshi_tansho_3a),
    ]

    tansho_result = np.zeros(fetcher.toroku_tosu)
    for atari in tansho_back:
        if not np.isnan(atari) and atari != 0:
            tansho_result[atari - 1] = 1.0

    tansho_result = tansho_result * fetcher.tansho_odds
    tansho_result = tansho_result - 1.0
    np.nan_to_num(tansho_result, copy=False)

    fukusho_result = np.zeros(fetcher.toroku_tosu)

    fukusho_1 = int_or_nan(back.haraimodoshi_fukusho_1a)

    if (not np.isnan(fukusho_1)) and fukusho_1 != 0:
        fukusho_result[fukusho_1 - 1] = (
            float(back.haraimodoshi_fukusho_1b) * 0.01
        )

    fukusho_2 = int_or_nan(back.haraimodoshi_fukusho_2a)

    if (not np.isnan(fukusho_2)) and fukusho_2 != 0:
        fukusho_result[fukusho_2 - 1] = (
            float(back.haraimodoshi_fukusho_2b) * 0.01
        )

    fukusho_3 = int_or_nan(back.haraimodoshi_fukusho_3a)
    if (not np.isnan(fukusho_3)) and fukusho_3 != 0:
        fukusho_result[fukusho_3 - 1] = (
            float(back.haraimodoshi_fukusho_3b) * 0.01
        )

    fukusho_4 = int_or_nan(back.haraimodoshi_fukusho_4a)
    if (not np.isnan(fukusho_4)) and fukusho_4 != 0:
        fukusho_result[fukusho_4 - 1] = (
            float(back.haraimodoshi_fukusho_4b) * 0.01
        )
    fukusho_5 = int_or_nan(back.haraimodoshi_fukusho_5a)
    if (not np.isnan(fukusho_5)) and fukusho_5 != 0:
        fukusho_result[fukusho_5 - 1] = (
            float(back.haraimodoshi_fukusho_5b) * 0.01
        )
    fukusho_result = fukusho_result - 1.0
    np.nan_to_num(fukusho_result, copy=False)
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
    umaren_result = np.zeros((fetcher.toroku_tosu, fetcher.toroku_tosu))

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
    umatan_result = np.zeros((fetcher.toroku_tosu, fetcher.toroku_tosu))

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
            sanrenpuku_result[atari[0] - 1, atari[1] - 1, atari[2] - 1] = 1.0

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
            sanrentan_result[atari[0] - 1, atari[1] - 1, atari[2] - 1] = 1.0
    sanrentan_result *= fetcher.sanrentan_odds
    sanrentan_result -= 1.0
    np.nan_to_num(sanrentan_result, copy=False)

    tansho_result *= tansho_buy
    fukusho_result *= fukusho_buy
    wide_result *= wide_buy
    umaren_result *= umaren_buy
    umatan_result *= umatan_buy
    sanrenpuku_result *= sanrenpuku_buy
    sanrentan_result *= sanrentan_buy

    tansho_benefit = np.sum(tansho_result)
    fukusho_benefit = np.sum(fukusho_result)
    wide_benefit = np.sum(wide_result)
    umaren_benefit = np.sum(umaren_result)
    umatan_benefit = np.sum(umatan_result)
    sanrenpuku_benefit = np.sum(sanrenpuku_result)
    sanrentan_benefit = np.sum(sanrentan_result)

    tansho_atari += np.count_nonzero(tansho_result > 0)
    fukusho_atari += np.count_nonzero(fukusho_result > 0.0)
    wide_atari += np.count_nonzero(wide_result > 0.0)
    umaren_atari += np.count_nonzero(umaren_result > 0.0)
    umatan_atari += np.count_nonzero(umatan_result > 0.0)
    sanrenpuku_atari += np.count_nonzero(sanrenpuku_result > 0.0)
    sanrentan_atari += np.count_nonzero(sanrentan_result > 0.0)

    tansho_sum += tansho_benefit
    fukusho_sum += fukusho_benefit
    wide_sum += wide_benefit
    umaren_sum += umaren_benefit
    umatan_sum += umatan_benefit
    sanrenpuku_sum += sanrenpuku_benefit
    sanrentan_sum += sanrentan_benefit
    race_sum[int(race.race_bango) - 1] += (
        sanrenpuku_benefit + sanrentan_benefit
    )
    shusso_tosu_sum[int(race.shusso_tosu) - 1] += (
        sanrenpuku_benefit + sanrentan_benefit
    )
    if tansho_benefit > 0:
        print(f"tansho: {tansho_benefit}")
        print(f"{race.kyori}m:第{race.race_bango}レース:頭数{race.shusso_tosu}")

    if fukusho_benefit > 0:
        print(f"fukusho: {fukusho_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    if wide_benefit > 0:
        print(f"wide: {wide_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    if umaren_benefit > 0:
        print(f"umaren: {umaren_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    if umatan_benefit > 0:
        print(f"umatan: {umatan_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    if sanrenpuku_benefit > 0:
        print(f"sanrenpuku: {sanrenpuku_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    if sanrentan_benefit > 0:
        print(f"sanrentan: {sanrentan_benefit}")
        print(f"{race.kyori}m:{race.race_bango}:{race.shusso_tosu}頭")

    tansho_graph.append(tansho_sum)
    fukusho_graph.append(fukusho_sum)
    wide_graph.append(wide_sum)
    umaren_graph.append(umaren_sum)
    umatan_graph.append(umatan_sum)
    sanrenpuku_graph.append(sanrenpuku_sum)
    sanrentan_graph.append(sanrentan_sum)

print("result")
print(f"tansho: {tansho_sum}")
print(f"fukusho: {fukusho_sum}")
print(f"wide: {wide_sum}")
print(f"umaren: {umaren_sum}")
print(f"umatan: {umatan_sum}")
print(f"sanrenpuku: {sanrenpuku_sum}")
print(f"sanrentan: {sanrentan_sum}")

print(f"tansho_bet: {tansho_bet}")
print(f"fukusho_bet: {fukusho_bet}")
print(f"wide_bet: {wide_bet}")
print(f"umaren_bet: {umaren_bet}")
print(f"umatan_bet: {umatan_bet}")
print(f"sanrenpuku_bet: {sanrenpuku_bet}")
print(f"sanrentan_bet: {sanrentan_bet}")


print(f"tansho_ratio: {(tansho_bet+tansho_sum)/tansho_bet}")
print(f"fukusho_ratio: {(fukusho_bet+fukusho_sum)/fukusho_bet}")
print(f"wide_ratio: {(wide_bet+wide_sum)/wide_bet}")
print(f"umaren_ratio: {(umaren_bet+umaren_sum)/umaren_bet}")
print(f"umatan_ratio: {(umatan_bet+umatan_sum)/umatan_bet}")
print(f"sanrenpuku_ratio: {(sanrenpuku_bet+sanrenpuku_sum)/sanrenpuku_bet}")
print(f"sanrentan_ratio: {(sanrentan_bet+sanrentan_sum)/sanrentan_bet}")

print(f"tansho_win_ratio: {tansho_atari/tansho_bet}")
print(f"fukusho_win_ratio: {fukusho_atari/fukusho_bet}")
print(f"wide_win_ratio: {wide_atari/wide_bet}")
print(f"umaren_win_ratio: {umaren_atari/umaren_bet}")
print(f"umatan_win_ratio: {umatan_atari/umatan_bet}")
print(f"sanrenpuku_win_ratio: {sanrenpuku_atari/sanrenpuku_bet}")
print(f"sanrentan_win_ratio: {sanrentan_atari/sanrentan_bet}")

end = time.time()
print(f"実行時間:{end-start}")

plt.plot(tansho_graph, label="tansho")
plt.plot(fukusho_graph, label="fukusho")
plt.plot(wide_graph, label="wide")
plt.plot(umaren_graph, label="umaren")
plt.plot(umatan_graph, label="umatan")
plt.plot(sanrenpuku_graph, label="sanrenpuku")
plt.plot(sanrentan_graph, label="sanrentan")
plt.legend()
plt.show()

for i, race_ratio in enumerate(shusso_tosu_sum):
    print(f"{i+1}レース:{race_ratio}")
plt.plot(shusso_tosu_sum, label="shusso_tosu_sum")
plt.plot(shusso_tosu_bet, label="shusso_tosu_bet")
plt.legend()
plt.show()

for i , race_ratio in enumerate(race_sum):
    print(f"第{i+1}レース:{race_ratio}")
plt.plot(race_sum,label="race_sum")
plt.plot(race_bet,label="race_bet")
plt.legend()
plt.show()