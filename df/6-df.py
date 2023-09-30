import pandas as pd
from uma_predict.db.models import Race, Career, Horse, Track
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy import desc
from utils import (
    horse_to_list,
    field_mapper,
    grade_mapper,
    roll_mapper,
    condition_mapper,
    soha_time_parser,
    horse_history,
    race_nyusen_result,
    hist_model_to_pandas,
    parse_time_sa,
)
import pprint
import random
import datetime
import numpy as np
import time

parse_start = time.time()
bataiju_mean = 469.12418300653593
bataiju_std = 28.81961143804636

pd.set_option("display.max_columns", 1000)
db = SessionLocal()
"""
column_list = ["keibajo","field","roll","field_condition","grade","distance","number_of_horse",
                "1_umaban","1_sex","1_age","1_bataiju","1_blinker","1_weight" ,
                "1_1_age","1_1_umaban","1_1_bataiju","1_1_blinker","1_1_weight","1_1_keibajo","1_1_field","1_1_roll","1_1_field_condition","1_1_grade","1_1_distance","1_1_number_of_horse","1_1_time","1_1_time_diff","1_1_last_3f","1_1_rank","1_1_3corner_rank","1_1_4corner_rank",
]
"""

race_condition = [
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

current_horse_columns = [
    "barei",
    "seibetsu_code",
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "tansho_odds",
]

horses_history_columns = [
    "barei",
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "babajotai_code_shiba",
    "babajotai_code_dirt",
    "track_code",
    "grade_code",
    "kyori",
    "shusso_tosu",
    "soha_time",
    "time_sa",
    "kohan_3f",
    "nyusen_juni",
    "corner_3",
    "corner_4",
    "tansho_odds",
    "tansho_ninkijun",
]

history_target_columns = [
    "barei",
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "condition",
    "field",
    "roll",
    "grade_code",
    "kyori",
    "shusso_tosu",
    "soha_time",
    "time_sa",
    "kohan_3f",
    "nyusen_juni",
    "corner_3",
    "corner_4",
    "tansho_odds",
    "tansho_ninkijun",
]

horses_hist_num_column = [
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "babajotai_code_shiba",
    "babajotai_code_dirt",
    "kyori",
    "shusso_tosu",
    "kohan_3f",
    "nyusen_juni",
    "corner_3",
    "corner_4",
    "tansho_odds",
    "tansho_ninkijun",
]

history_number = 5
predict_rank = 3


horse_history_index = [
    histindex + umaban * 10
    for umaban in range(1, 19)
    for histindex in range(1, history_number + 1)
]


data = pd.DataFrame()

races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen == "2022",        
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code <= "26",
        Race.track_code >= "00",
        Race.nyusen_tosu > "03",
        Race.kyoso_joken_code != "701",
    )
).all()

for race in races:
    pprint.pprint(race.__dict__)
    horses = race_nyusen_result(race, db)

    nyusen_umaban_list = [0] * 18

    for horse in horses:
        nyusen_umaban_list[int(horse.nyusen_juni) - 1] = int(horse.umaban)

    horse_current = pd.DataFrame(
        0, columns=current_horse_columns, index=list(range(1, 19))
    )
    horse_current["tansho_odds"] = 10000
    
    horses_history = pd.DataFrame(
        index=[horse_history_index],
    )
    for horse in horses:
        horse_master = db.get(Horse, horse.ketto_toroku_bango)

        horse_current.loc[int(horse.umaban)] = horse_to_list(
            horse, horse_master.seinengappi
        )

        horse_hist = horse_history(
            horse, db, end=horse.kaisai_nen + horse.kaisai_tsukihi
        )

        hist_len = len(horse_hist)
        hist_index = 1

        for hist in horse_hist:
            past_race = hist[1]

            if past_race.track_code > "26" or past_race.track_code < "00":
                continue

            hist_dict = {
                "barei": (
                    datetime.datetime.strptime(
                        hist[0].kaisai_nen + hist[0].kaisai_tsukihi, "%Y%m%d"
                    )
                    - datetime.datetime.strptime(
                        horse_master.seinengappi, "%Y%m%d"
                    )
                ).days,
                "bataiju": float(hist[0].bataiju),
                "blinker_shiyo_kubun": int(hist[0].blinker_shiyo_kubun),
                "futan_juryo": float(hist[0].futan_juryo),
                "condition": int(hist[1].babajotai_code_shiba)
                + int(hist[1].babajotai_code_dirt),
                "field": field_mapper(hist[1].track_code),
                "roll": roll_mapper(hist[1].track_code),
                "grade_code": grade_mapper(hist[1].grade_code),
                "kyori": float(hist[1].kyori),
                "shusso_tosu": float(hist[1].shusso_tosu),
                "soha_time": soha_time_parser(hist[0].soha_time),
                "time_sa": parse_time_sa(hist[0].time_sa),
                "kohan_3f": float(hist[0].kohan_3f),
                "nyusen_juni": int(hist[0].nyusen_juni),
                "corner_3": int(hist[0].corner_3),
                "corner_4": int(hist[0].corner_4),
                "tansho_odds": float(hist[0].tansho_odds),
                "tansho_ninkijun": int(hist[0].tansho_ninkijun),
            }

            horses_history.loc[
                hist_index + int(horse.umaban) * 10, hist_dict.keys()
            ] = hist_dict.values()
            if hist_index == history_number:
                break
            hist_index += 1
        # 過去のレースが少ない場合は、最後のレースをコピーする
        if hist_len >= 1 and hist_len < history_number:
            horses_history.loc[
                int(horse.umaban) * 10
                + 1
                + hist_len : int(horse.umaban) * 10
                + 1
                + history_number
            ] = horses_history.loc[int(horse.umaban) * 10 + hist_len].values

    horses_history = horses_history[history_target_columns]

    # スケーリング

    normalize_target = [
        "soha_time",
        "time_sa",
        "kohan_3f",
    ]

    rank_target = [
        "shusso_tosu",
        "nyusen_juni",
        "corner_3",
        "corner_4",
        "tansho_ninkijun",
    ]

    normarlize_target_min = horses_history[normalize_target].min()
    horses_history[normalize_target] = (
        horses_history[normalize_target] - normarlize_target_min
    ) / (horses_history[normalize_target].max() - normarlize_target_min)

    horses_history[rank_target] = 1/horses_history[rank_target]
    horses_history["barei"] = horses_history["barei"] / (8 * 365)
    horses_history["kyori"] = (horses_history["kyori"] - 1000.0) / 2600.0
    horses_history["bataiju"] = (horses_history["bataiju"] - bataiju_mean) / (
        bataiju_std
    )
    horses_history["futan_juryo"] = (
        horses_history["futan_juryo"] - 510.0
    ) / 90.0

    for rank_column in rank_target:
        horses_history.fillna({rank_column: 0}, inplace=True)
    horses_history.fillna({"tansho_odds": 10000}, inplace=True)
    horses_history.fillna(0, inplace=True)
    horses_history["tansho_odds"] = 10 / horses_history["tansho_odds"]

    row_index = int(
        race.kaisai_nen
        + race.kaisai_tsukihi
        + race.keibajo_code
        + race.race_bango
    )

    result_row = pd.DataFrame(
        [
            [
                field_mapper(race.track_code),
                roll_mapper(race.track_code),
                condition_mapper(race),
                grade_mapper(race.grade_code),
                (int(race.kyori) - 1000.0) / 2600.0,
                int(race.shusso_tosu) / 18.0,
            ]
        ],
        columns=race_condition,
        index=[row_index],
    )

    for i in range(1, 19):
        uma_column = [
            str(i) + "_" + column for column in current_horse_columns
        ]
        current_uma = pd.DataFrame(
            [horse_current.loc[i].values],
            index=[row_index],
            columns=uma_column,
        )
        current_uma[str(i) + "_" + "barei"] = current_uma[
            str(i) + "_" + "barei"
        ] / (8 * 365)
        current_uma[str(i) + "_" + "bataiju"] = (
            current_uma[str(i) + "_" + "bataiju"] - bataiju_mean
        ) / bataiju_std
        current_uma[str(i) + "_" + "futan_juryo"] = (
            current_uma[str(i) + "_" + "futan_juryo"] - 510.0
        ) / 90.0
        current_uma[str(i) + "_" + "tansho_odds"] = (
            1 / current_uma[str(i) + "_" + "tansho_odds"]
        )
        result_row = pd.concat([result_row, current_uma], axis=1)
        for j in range(1, history_number + 1):
            hist_column = [
                str(i) + "_" + str(j) + "_" + column
                for column in horses_history.columns.values
            ]
            hist_uma = pd.DataFrame(
                horses_history.loc[i * 10 + j].values,
                index=[row_index],
                columns=hist_column,
            )
            result_row = pd.concat([result_row, hist_uma], axis=1)

    target = pd.DataFrame(
        [nyusen_umaban_list[0:3]],
        columns=["first", "second", "third"],
        index=[row_index],
    )

    result_row = pd.concat([result_row, target], axis=1)
    data = pd.concat([data, result_row], axis=0)

pprint.pprint(data)
print(data.isnull().values.sum())
print(data.isinf().values.sum())
time_end = time.time()
print(time_end - parse_start)
target_year = "2022"
data.to_csv("./data6/" + target_year + "_" + str(history_number) + ".csv")
