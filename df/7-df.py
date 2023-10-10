import pandas as pd
from db.models import Race, Career, Horse, Track
from db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy import desc
from utils import (
    horse_to_list,
    field_mapper,
    grade_mapper,
    roll_mapper,
    condition_mapper,
    soha_time_parser,
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

race_condition = [
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

current_horse_column_master = [
    "barei" "seibetu",
    "age",
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "tansho_odds",
    "tansho_ninkijun",
]

horse_hist_column_master = [
    "barei",
    "bataiju",
    "blinker_shiyo_kubun",
    "futan_juryo",
    "condition" "field",
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


hist_number = 5
predict_rank = 3

race_df_columns = []
race_df_columns.append(current_horse_column_master)
for i in range(1, hist_number + 1):
    hist_column = [column + str(i) for column in horse_hist_column_master]
    race_df_columns.append(hist_column)


horse_history_index = [
    histindex + umaban * 10
    for umaban in range(1, 19)
    for histindex in range(1, history_number + 1)
]


data = pd.DataFrame()

races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen <= "2021",
        Race.kaisai_nen >= "2012",
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
    horses = db.scalars(
        select(Career)
        .filter(
            Career.kaisai_nen == race.kaisai_nen,
            Career.kaisai_tsukihi == race.kaisai_tsukihi,
            Career.keibajo_code == race.keibajo_code,
            Career.race_bango == race.race_bango,
            Career.ijo_kubun_code != "1",
            Career.ijo_kubun_code != "2",
            Career.ijo_kubun_code != "3",
            Career.ijo_kubun_code != "4",
        )
        .order_by(Career.nyusen_juni)
    ).all()
    nyusen_umaban_list = [0] * 18

    for horse in horses:
        nyusen_umaban_list[int(horse.nyusen_juni) - 1] = int(horse.umaban)

    race_df = pd.DataFrame(index=list(range(1, 19)))

    for horse in horses:
        horse_master = db.get(Horse, horse.ketto_toroku_bango)
        current_horse_dict = {
            "barei": (
                datetime.datetime.strptime(
                    horse.kaisai_nen + horse.kaisai_tsukihi, "%Y%m%d"
                )
                - datetime.datetime.strptime(
                    horse_master.seinengappi, "%Y%m%d"
                )
            ).days,
            "seibetu": 0
            if horse.seibetsu_code == "2"
            else 1
            if horse.seibetsu_code == "1"
            else 2,
            "bataiju": int(horse.bataiju),
            "blinker_shiyo_kubun": int(horse.blinker_shiyo_kubun),
            "futan_juryo": int(horse.futan_juryo),
            "tansho_odds": float(horse.tansho_odds),
            "tansho_ninkijun": int(horse.tansho_ninkijun),
        }

        race_df.loc[int(horse.umaban)] = current_horse_dict.values()

        horse_hist = db.scalars(
            select(Career)
            .filter(
                Career.ketto_toroku_bango == horse.ketto_toroku_bango,
                Career.kaisai_nen + Career.kaisai_tsukihi
                < horse.kaisai_nen + horse.kaisai_tsukihi,
                Career.kaisai_nen + Career.kaisai_tsukihi >= "20020615",
                Career.keibajo_code <= "10",
                Career.keibajo_code >= "01",
                Career.ijo_kubun_code != "1",
                Career.ijo_kubun_code != "2",
                Career.ijo_kubun_code != "3",
                Career.ijo_kubun_code != "4",
                Career.nyusen_juni != "00",
            )
            .order_by(desc(Career.kaisai_nen + Career.kaisai_tsukihi))
        ).all()

        hist_len = len(horse_hist)
        hist_index = 1
        for hist in horse_hist:
            past_race = db.scalars(
                select(Race).filter(
                    Race.kaisai_nen == hist.kaisai_nen,
                    Race.kaisai_tsukihi == hist.kaisai_tsukihi,
                    Race.keibajo_code == hist.keibajo_code,
                    Race.race_bango == hist.race_bango,
                    Race.track_code <= "26",
                    Race.track_code >= "00",
                )
            ).first()

            if past_race is None:
                continue
            past_race_condition = int(past_race.babajotai_code_shiba) + int(
                past_race.babajotai_code_dirt
            )

            time_mean = db.scalars(
                select(Track).filter(
                    Track.keibajo_code == past_race.keibajo_code,
                    Track.start_year <= past_race.kaisai_nen,
                    Track.end_year >= past_race.kaisai_nen,
                    Track.kyori == past_race.kyori,
                    Track.track_code == past_race.track_code,
                    Track.babajotai_code == past_race_condition,
                    Track.is_last3f == False,
                )
            ).first()
            last3f_mean = db.scalars(
                select(Track).filter(
                    Track.keibajo_code == hist.keibajo_code,
                    Track.start_year <= hist.kaisai_nen,
                    Track.end_year >= hist.kaisai_nen,
                    Track.kyori == past_race.kyori,
                    Track.track_code == past_race.track_code,
                    Track.babajotai_code == past_race_condition,
                    Track.is_last3f == True,
                )
            ).first()

            hist_dict = {
                "barei": (
                    (
                        datetime.datetime.strptime(
                            hist.kaisai_nen + hist.kaisai_tsukihi, "%Y%m%d"
                        )
                        - datetime.datetime.strptime(
                            horse_master.seinengappi, "%Y%m%d"
                        )
                    ).days
                )
                / (8 * 365),
                "bataiju": 4
                + (float(hist.bataiju) - bataiju_mean) / bataiju_std,
                "blinker_shiyo_kubun": int(hist.blinker_shiyo_kubun),
                "futan_juryo": (float(hist.futan_juryo) - 510.0) / 90.0,
                "condition": past_race_condition,
                "field": field_mapper(past_race.track_code),
                "roll": roll_mapper(past_race.track_code),
                "grade_code": grade_mapper(past_race.grade_code),
                "kyori": (float(past_race.kyori) - 1000.0) / 2600.0,
                "shusso_tosu": float(past_race.shusso_tosu) / 18,
                "soha_time": 4
                - (
                    (soha_time_parser(hist.soha_time) - time_mean.mean)
                    / time_mean.std
                ),
                "time_sa": float(hist.time_sa[1:]) * 0.1
                if hist.time_sa[0] == "+"
                else 0.0,
                "kohan_3f": 1
                - (
                    (float(hist.kohan_3f) * 0.1 - last3f_mean.mean)
                    / last3f_mean.std
                ),
                "nyusen_juni": 1 / int(hist.nyusen_juni),
                "corner_3": 1 / int(hist.corner_3),
                "corner_4": 1 / int(hist.corner_4),
                "tansho_odds": 10 / float(hist.tansho_odds),
                "tansho_ninkijun": 1 / int(hist.tansho_ninkijun),
            }
            hist_dict_key = [
                str(hist_index) + keys for keys in hist_dict.keys()
            ]
            race_df.loc[int(horse.umaban), hist_dict_key] = hist_dict.values()

            if hist_index == history_number:
                break
            hist_index += 1

    # Nanを埋める

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
            horse_current[horse_current["umaban"] == i].values,
            index=[row_index],
            columns=uma_column,
        )
        current_uma[str(i) + "_" + "age"] = current_uma[
            str(i) + "_" + "age"
        ] / (8 * 365)
        current_uma[str(i) + "_" + "umaban"] = (
            current_uma[str(i) + "_" + "umaban"] / 18.0
        )
        current_uma[str(i) + "_" + "bataiju"] = (
            current_uma[str(i) + "_" + "bataiju"] - bataiju_mean
        ) / bataiju_std
        current_uma[str(i) + "_" + "weight"] = (
            current_uma[str(i) + "_" + "weight"] - 510.0
        ) / 90.0
        result_row = pd.concat([result_row, current_uma], axis=1)
        for j in range(1, history_number + 1):
            hist_column = [
                str(i) + "_" + str(j) + "_" + column
                for column in horses_history_columns
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
time_end = time.time()
print(time_end - parse_start)

pprint.pprint(data)
print(data.isnull().values.sum())
target_year = "2012_2021"
data.to_csv("./data4/" + target_year + "_" + str(history_number) + ".csv")
