import pandas as pd
from uma_predict.db.models import Race, Career, Horse, Track
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy import desc
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
column_name = [
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

race_condition = [
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

current_horse_columns = [
    "umaban",
    "sex",
    "age",
    "bataiju",
    "blinker",
    "weight",
]

horses_history_columns = [
    "age",
    "umaban",
    "bataiju",
    "blinker",
    "weight",
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
    "time",
    "time_diff",
    "last_3f",
    "rank",
    "3corner_rank",
    "4corner_rank",
]


history_number = 4
predict_rank = 3


def horse_to_list(horse, seinengappi):
    horse = [
        int(horse.umaban),
        int(horse.seibetsu_code),
        (
            datetime.datetime.strptime(
                horse.kaisai_nen + horse.kaisai_tsukihi, "%Y%m%d"
            )
            - datetime.datetime.strptime(seinengappi, "%Y%m%d")
        ).days,
        int(horse.bataiju),
        int(horse.blinker_shiyo_kubun),
        float(horse.futan_juryo),
    ]
    return horse


def field_mapper(track_code: str):
    return 0 if track_code >= "23" else 1


def grade_mapper(grade: str):
    if grade == "_":
        return 0
    elif grade == "E":
        return 1
    elif grade == "D":
        return 2
    elif grade == "L":
        return 3
    elif grade == "C":
        return 4
    elif grade == "B":
        return 5
    elif grade == "A":
        return 6
    else:
        return 0


def roll_mapper(track_code: str):
    return (
        1
        if track_code == "11" or track_code == "12" or track_code == "23"
        else -1
        if track_code == "00"
        else 0
    )


def condition_mapper(race: Race):
    return (
        int(race.babajotai_code_dirt)
        if race.track_code == "23" or race.track_code == "24"
        else int(race.babajotai_code_shiba)
    )


def soha_time_parser(soha_time: str):
    return float(soha_time[0]) * 60 + float(soha_time[1:]) * 0.1


for i in range(1, 19):
    current_horse_info = [
        str(i) + "_umaban",
        str(i) + "_sex",
        str(i) + "_age",
        str(i) + "_bataiju",
        str(i) + "_blinker",
        str(i) + "_weight",
    ]
    column_name.extend(current_horse_info)
    for j in range(1, history_number + 1):
        history = [
            str(i) + "_" + str(j) + "_age",  # 8*365で割る
            str(i) + "_" + str(j) + "_umaban",  # 18で割る
            str(i) + "_" + str(j) + "_bataiju",  # 全体で標準化
            str(i) + "_" + str(j) + "_blinker",
            str(i) + "_" + str(j) + "_weight",  # 全体で正規化
            str(i) + "_" + str(j) + "_field",
            str(i) + "_" + str(j) + "_roll",
            str(i) + "_" + str(j) + "_field_condition",
            str(i) + "_" + str(j) + "_grade",
            str(i) + "_" + str(j) + "_distance",  # 正規化
            str(i) + "_" + str(j) + "_number_of_horse",  # 18で割る
            str(i) + "_" + str(j) + "_time",  # コースごとに標準化
            str(i) + "_" + str(j) + "_time_diff",  # そのまま使う
            str(i) + "_" + str(j) + "_last_3f",  # コースごとに標準化
            str(i) + "_" + str(j) + "_rank",
            str(i) + "_" + str(j) + "_3corner_rank",
            str(i) + "_" + str(j) + "_4corner_rank",
        ]
        column_name.extend(history)


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

    horse_current = pd.DataFrame(
        columns=current_horse_columns, index=list(range(1, 19))
    )
    bottom_horses_number = int(race.nyusen_tosu) // 5 + 1
    bottom_horses = list(
        filter(
            lambda x: int(x.nyusen_juni)
            >= int(race.nyusen_tosu) - bottom_horses_number + 1,
            horses,
        )
    )

    horses_history = pd.DataFrame(
        columns=horses_history_columns, index=[horse_history_index]
    )
    for horse in horses:
        horse_master = db.get(Horse, horse.ketto_toroku_bango)

        horse_current.loc[int(horse.umaban)] = horse_to_list(
            horse, horse_master.seinengappi
        )

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
            past_race_condition = (
                past_race.babajotai_code_dirt
                if past_race.track_code == "23" or past_race.track_code == "24"
                else past_race.babajotai_code_shiba
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

            hist_row = [
                (
                    datetime.datetime.strptime(
                        hist.kaisai_nen + hist.kaisai_tsukihi, "%Y%m%d"
                    )
                    - datetime.datetime.strptime(
                        horse_master.seinengappi, "%Y%m%d"
                    )
                ).days,
                int(hist.umaban),
                int(hist.bataiju),
                int(hist.blinker_shiyo_kubun),
                float(hist.futan_juryo),
                field_mapper(past_race.track_code),
                roll_mapper(past_race.track_code),
                condition_mapper(past_race),
                grade_mapper(past_race.grade_code),
                int(past_race.kyori),
                int(past_race.shusso_tosu),
                (soha_time_parser(hist.soha_time) - time_mean.mean)
                / time_mean.std,
                float(hist.time_sa[1:]) * 0.1
                if hist.time_sa[0] == "+"
                else 0.0,
                (float(hist.kohan_3f) * 0.1 - last3f_mean.mean)
                / last3f_mean.std,
                int(hist.nyusen_juni),
                int(hist.corner_3),
                int(hist.corner_4),
            ]
            horses_history.loc[hist_index + int(horse.umaban) * 10] = hist_row
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
    # Nanを埋める
    for row in horse_current[horse_current["umaban"].isnull()].itertuples():
        bottom_horse = bottom_horses[random.randint(0, len(bottom_horses) - 1)]
        horse_current.loc[row.Index] = horse_current[
            horse_current["umaban"] == int(bottom_horse.umaban)
        ].values
        horse_current.at[row.Index, "umaban"] = row.Index
        horses_history.loc[
            1 + row.Index * 10 : 1 + row.Index * 10 + history_number,
            horses_history_columns,
        ] = horses_history.loc[
            1
            + int(bottom_horse.umaban) * 10 : 1
            + int(bottom_horse.umaban) * 10
            + history_number,
            horses_history_columns,
        ].values
    horses_history.fillna(
        {
            "age": horses_history["age"].mean(),
            "umaban": horses_history["umaban"].mode().iloc[0],
            "bataiju": horses_history["bataiju"].mean(),
            "blinker": horses_history["blinker"].mode().iloc[0],
            "weight": horses_history["weight"].mode().iloc[0],
            "field": horses_history["field"].mode().iloc[0],
            "roll": horses_history["roll"].mode().iloc[0],
            "field_condition": horses_history["field_condition"]
            .mode()
            .iloc[0],
            "grade": horses_history["grade"].mode().iloc[0],
            "distance": horses_history["distance"].max(),
            "number_of_horse": 18,
            "time": horses_history["time"].max(),
            "time_diff": horses_history["time_diff"].mean(),
            "last_3f": horses_history["last_3f"].max(),
            "rank": random.randint(4, 18),
            "3corner_rank": random.randint(4, 18),
            "4corner_rank": random.randint(4, 18),
        },
        inplace=True,
    )
    # スケーリング
    horses_history["age"] = horses_history["age"] / (8 * 365)
    horses_history["umaban"] = horses_history["umaban"] / 18
    horses_history["bataiju"] = (
        horses_history["bataiju"] - bataiju_mean
    ) / bataiju_std
    horses_history["grade"] = horses_history["grade"] / 6
    horses_history["number_of_horse"] = horses_history["number_of_horse"] / 18

    horses_history["weight"] = (horses_history["weight"] - 510.0) / 90.0
    horses_history["distance"] = horses_history["distance"] / 200
    horses_history["rank"] = horses_history["rank"] / 18
    horses_history["3corner_rank"] = horses_history["3corner_rank"] / 18
    horses_history["4corner_rank"] = horses_history["4corner_rank"] / 18
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
target_year = "2022"
data.to_csv("./data3/" + target_year + ".csv")
