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
import matplotlib.pyplot as plt

start = time.time()
db = SessionLocal()
np.set_printoptions(linewidth=np.inf)


train_data = pl.read_parquet("df\dataOdds\\2016-2021-30_v3.parquet")
column_len = train_data.shape[1] - 18

model = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_119/checkpoints/epoch=59-step=29160.ckpt",
    input_size=column_len,
    output_size=18,
    hiden_layer_size=[1024, 512, 32],
)
model.eval()

train_data_2 = pl.read_parquet("df\dataEt\\2016-2021-0-10.parquet")
model_2 = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_107_h0/checkpoints/epoch=59-step=8580.ckpt",
    input_size=train_data_2.shape[1] - 18,
    output_size=18,
)
model_2.eval()

train_data_3 = pl.read_parquet("df\dataOdds\\2016-2021-30_v3.parquet")
column_len = train_data_3.shape[1] - 18

model_3 = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_123/checkpoints/epoch=119-step=29160.ckpt",
    input_size=column_len,
    output_size=18,
    hiden_layer_size=[1024, 512, 256, 32],
)
model_3.eval()


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
        Race.nyusen_tosu > "08",
        Race.kyoso_joken_code != "701",
    )
).all()

model_val = 0
model_2_val = 0
model_3_val = 0


fuku_model_val = [0] * 8
fuku_model_2_val = [0] * 8
fuku_model_3_val = [0] * 8


fukusho_all = 0

prob_th = 0.02
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

    fetcher.setup_odds_df_db()
    odds_data = fetcher.odds_df
    odds_data = torch.from_numpy(odds_data.to_numpy()).float()

    fetcher.toroku_tosu = int(race.toroku_tosu)
    _y = F.softmax(model(odds_data), dim=1)[:, : fetcher.toroku_tosu]

    y = _y.detach().numpy().copy()[0]
    y = y / y.sum()
    fetcher.get_horse_data_from_db(0)
    _y_2 = F.softmax(model_2(fetcher.horse_data[:, :-18]), dim=1)[
        :, : fetcher.toroku_tosu
    ]

    y_2 = _y_2.detach().numpy().copy()[0]
    y_2 = y_2 / y_2.sum()

    fetcher.get_horse_data_from_db(2)
    _y_3 = F.softmax(model_3(odds_data), dim=1)[:, : fetcher.toroku_tosu]

    y_3 = _y_3.detach().numpy().copy()[0]
    y_3 = y_3 / y_3.sum()

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

    fukusho_result = np.zeros(fetcher.toroku_tosu)

    fukusho_1 = int_or_nan(back.haraimodoshi_fukusho_1a)

    if (not np.isnan(fukusho_1)) and fukusho_1 != 0:
        fukusho_result[fukusho_1 - 1] = 1

    fukusho_2 = int_or_nan(back.haraimodoshi_fukusho_2a)

    if (not np.isnan(fukusho_2)) and fukusho_2 != 0:
        fukusho_result[fukusho_2 - 1] = 1

    fukusho_3 = int_or_nan(back.haraimodoshi_fukusho_3a)
    if (not np.isnan(fukusho_3)) and fukusho_3 != 0:
        fukusho_result[fukusho_3 - 1] = 1

    fukusho_4 = int_or_nan(back.haraimodoshi_fukusho_4a)
    if (not np.isnan(fukusho_4)) and fukusho_4 != 0:
        fukusho_result[fukusho_4 - 1] = 1

    fukusho_5 = int_or_nan(back.haraimodoshi_fukusho_5a)
    if (not np.isnan(fukusho_5)) and fukusho_5 != 0:
        fukusho_result[fukusho_5 - 1] = 1

    np.nan_to_num(fukusho_result, copy=False)
    print(np.sum(fukusho_result))
    fukusho_all += np.sum(fukusho_result)

    for i in range(0, 8):
        top_number = i + 1
        y_top_3 = sorted(y.ravel())[-top_number]
        y_2_top_3 = sorted(y_2.ravel())[-top_number]
        y_3_top_3 = sorted(y_3.ravel())[-top_number]

        y_fuku = np.where(y >= y_top_3, 1, 0)
        y_2fuku = np.where(y_2 >= y_2_top_3, 1, 0)
        y_3fuku = np.where(y_3 >= y_3_top_3, 1, 0)

        fuku_model_val[i] += np.sum(y_fuku * fukusho_result)
        fuku_model_2_val[i] += np.sum(y_2fuku * fukusho_result)
        fuku_model_3_val[i] += np.sum(y_3fuku * fukusho_result)

    for atari in tansho_back:
        if not np.isnan(atari) and atari != 0:
            y[atari - 1] = 0.0
            y_2[atari - 1] = 0.0
            y_3[atari - 1] = 0.0
    
    y = y / y.sum()
    y_2 = y_2 / y_2.sum()
    y_3 = y_3 / y_3.sum()

    model_val += np.sum(y * fukusho_result)
    model_2_val += np.sum(y_2 * fukusho_result)
    model_3_val += np.sum(y_3 * fukusho_result)

    print(race.kaisai_nen + race.kaisai_tsukihi)

fuku_model_val = np.array(fuku_model_val)
fuku_model_2_val = np.array(fuku_model_2_val)
fuku_model_3_val = np.array(fuku_model_3_val)

fuku_model_val /= fukusho_all
fuku_model_2_val /= fukusho_all
fuku_model_3_val /= fukusho_all


print(f"model_val:{model_val}")
print(f"model_2_val:{model_2_val}")
print(f"model_3_val:{model_3_val}")


print(f"fuku_model_val:{fuku_model_val}")
print(f"fuku_model_2_val:{fuku_model_2_val}")
print(f"fuku_model_3_val:{fuku_model_3_val}")

end_time = time.time()
print(f"end_time:{end_time-start}")

plt.plot(fuku_model_val, label="fuku_model_val")
plt.plot(fuku_model_2_val, label="fuku_model_2_val")
plt.plot(fuku_model_3_val, label="fuku_model_3_val")

plt.legend()
plt.show()
