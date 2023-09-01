import pandas as pd
from uma_predict.db.models import Race, Career, Horse
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy import desc
import pprint
import random
import datetime
import numpy as np
import time

parse_start = time.time()

pd.set_option("display.max_columns", 1000)
db = SessionLocal()
"""
column_list = ["keibajo","field","roll","field_condition","grade","distance","number_of_horse",
                "1_umaban","1_sex","1_age","1_bataiju","1_blinker","1_weight" ,
                "1_1_age","1_1_umaban","1_1_bataiju","1_1_blinker","1_1_weight","1_1_keibajo","1_1_field","1_1_roll","1_1_field_condition","1_1_grade","1_1_distance","1_1_number_of_horse","1_1_time","1_1_time_diff","1_1_last_3f","1_1_rank","1_1_3corner_rank","1_1_4corner_rank",
]
"""
column_name = [
    "keibajo",
    "field",
    "roll",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

race_condition = [
    "keibajo",
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
    "keibajo",
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


history_number = 3
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
        ).days*0.01,
        int(horse.bataiju)*0.02,
        int(horse.blinker_shiyo_kubun),
        float(horse.futan_juryo) * 0.002,
    ]
    return horse


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
            str(i) + "_" + str(j) + "_age",
            str(i) + "_" + str(j) + "_umaban",
            str(i) + "_" + str(j) + "_bataiju",
            str(i) + "_" + str(j) + "_blinker",
            str(i) + "_" + str(j) + "_weight",
            str(i) + "_" + str(j) + "_keibajo",
            str(i) + "_" + str(j) + "_field",
            str(i) + "_" + str(j) + "_roll",
            str(i) + "_" + str(j) + "_field_condition",
            str(i) + "_" + str(j) + "_grade",
            str(i) + "_" + str(j) + "_distance",
            str(i) + "_" + str(j) + "_number_of_horse",
            str(i) + "_" + str(j) + "_time",
            str(i) + "_" + str(j) + "_time_diff",
            str(i) + "_" + str(j) + "_last_3f",
            str(i) + "_" + str(j) + "_rank",
            str(i) + "_" + str(j) + "_3corner_rank",
            str(i) + "_" + str(j) + "_4corner_rank",
        ]
        column_name.extend(history)
result_column_name = []

for i in range(1, 19):
    for j in range(1, predict_rank + 1):
        result_column_name.append("result_" + str(i) + "_" + str(j))

column_name.extend(result_column_name)


horse_history_index = [
    histindex + umaban * 10
    for umaban in range(1, 19)
    for histindex in range(1, history_number + 1)
]


def chakusa_num(chakusa_list: list):
    chakusa_num = [1.0] * 17
    for index, chakusa in enumerate(chakusa_list):
        if chakusa == "D  ":
            chakusa_num[index] = 0.5
        elif chakusa == "H  ":
            chakusa_num[index] = 0.525
        elif chakusa == "A  ":
            chakusa_num[index] = 0.55
        elif chakusa == "K  ":
            chakusa_num[index] = 0.6
        elif chakusa == " 12":
            chakusa_num[index] = 0.691
        elif chakusa == " 34":
            chakusa_num[index] = 0.773
        elif chakusa == "1  ":
            chakusa_num[index] = 0.841
        elif chakusa == "112":
            chakusa_num[index] = 0.933
        elif chakusa == "114":
            chakusa_num[index] = 0.894
        elif chakusa == "134":
            chakusa_num[index] = 0.956
        elif chakusa == "2  ":
            chakusa_num[index] = 0.977
        elif chakusa == "212":
            chakusa_num[index] = 0.993
    return chakusa_num


def rank_probability(
    rank_list: list, chakusa_num: list, limit: int, index_name, column_name
):
    prob_mat = np.zeros((limit + 1, limit + 1))
    prob_mat[0, 0] = 1
    for i in range(0, predict_rank):
        for j in range(i, limit):
            prob_mat[j + 1, i] = prob_mat[j, i] * (1 - chakusa_num[j])
            prob_mat[j, i] = prob_mat[j, i] * chakusa_num[j]
        prob_mat[i, i + 1 :] = prob_mat[i + 1 :, i]
        prob_mat[i + 1, i + 1] = 1 - prob_mat[i + 1, : i + 1].sum()

    result = pd.DataFrame(0.0, columns=column_name, index=[index_name])
    for i in range(0, limit):
        for j in range(1, predict_rank + 1):
            result.at[
                index_name, "result_" + str(rank_list[i]) + "_" + str(j)
            ] = prob_mat[i, j - 1]
    return result




target_year = "2006-2021"

data = pd.DataFrame()

races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen<="2021",
        Race.kaisai_nen>="2006",
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
    chakusa_list = ["   "] * 17
    nyusen_umaban_list = [0] * 18

    for horse in horses:
        nyusen_umaban_list[int(horse.nyusen_juni) - 1] = int(horse.umaban)
        if horse.chakusa_code_1 != "   ":
            chakusa_list[int(horse.nyusen_juni) - 2] = horse.chakusa_code_1
            if horse.chakusa_code_2 != "   ":
                chakusa_list[int(horse.nyusen_juni) - 3] = horse.chakusa_code_2
                if horse.chakusa_code_3 != "   ":
                    chakusa_list[
                        int(horse.nyusen_juni) - 4
                    ] = horse.chakusa_code_3

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
            .limit(history_number)
        ).all()

        hist_len = len(horse_hist)
        for hist_index, hist in enumerate(horse_hist):
            past_race = db.scalars(
                select(Race).filter(
                    Race.kaisai_nen == hist.kaisai_nen,
                    Race.kaisai_tsukihi == hist.kaisai_tsukihi,
                    Race.keibajo_code == hist.keibajo_code,
                    Race.race_bango == hist.race_bango,
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
                ).days*0.01,
                int(hist.umaban),
                int(hist.bataiju)*0.02,
                int(hist.blinker_shiyo_kubun),
                float(hist.futan_juryo) * 0.002,
                int(hist.keibajo_code),
                0 if past_race.track_code >= "23" else 1,
                1
                if past_race.track_code == "11"
                or past_race.track_code == "12"
                or past_race.track_code == "23"
                else 0
                if past_race.track_code == "00"
                else -1,
                int(past_race.babajotai_code_dirt)
                if past_race.track_code == "23" or past_race.track_code == "24"
                else int(past_race.babajotai_code_shiba),
                0
                if past_race.grade_code == "_"
                else 1
                if past_race.grade_code == "E"
                else 2
                if past_race.grade_code == "D"
                else 3
                if past_race.grade_code == "L"
                else 4
                if past_race.grade_code == "C"
                else 5
                if past_race.grade_code == "B"
                else 6
                if past_race.grade_code == "A"
                else 0,
                int(past_race.kyori)/200,
                int(past_race.shusso_tosu),
                float(hist.soha_time[0]) * 60
                + float(hist.soha_time[1:]) * 0.1,
                float(hist.time_sa[1:]) * 0.1
                if hist.time_sa[0] == "+"
                else 0.0,
                float(hist.kohan_3f) * 0.1,
                int(hist.nyusen_juni),
                int(hist.corner_3),
                int(hist.corner_4),
            ]
            horses_history.loc[
                hist_index + int(horse.umaban) * 10 + 1
            ] = hist_row

        if hist_len >= 1 and hist_len < history_number:
            horses_history.loc[
                int(horse.umaban) * 10
                + 1
                + hist_len : int(horse.umaban) * 10
                + 1
                + history_number
            ] = horses_history.loc[int(horse.umaban) * 10 + hist_len].values

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
            "keibajo": horses_history["keibajo"].mode().iloc[0],
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

    row_index = int(
        race.kaisai_nen
        + race.kaisai_tsukihi
        + race.keibajo_code
        + race.race_bango
    )

    result_row = pd.DataFrame(
        [
            [
                int(hist.keibajo_code),
                0 if race.track_code >= "23" else 1,
                1
                if race.track_code == "11"
                or race.track_code == "12"
                or race.track_code == "23"
                else 0
                if race.track_code == "00"
                else -1,
                int(race.babajotai_code_dirt)
                if race.track_code == "23" or race.track_code == "24"
                else int(race.babajotai_code_shiba),
                0
                if race.grade_code == "_"
                else 1
                if race.grade_code == "E"
                else 2
                if race.grade_code == "D"
                else 3
                if race.grade_code == "L"
                else 4
                if race.grade_code == "C"
                else 5
                if race.grade_code == "B"
                else 6
                if race.grade_code == "A"
                else 0,
                int(race.kyori)/200,
                int(race.shusso_tosu),
            ]
        ],
        columns=race_condition,
        index=[row_index],
    )

    result_prob = rank_probability(
        nyusen_umaban_list,
        chakusa_num(chakusa_list),
        min([6, int(race.nyusen_tosu) - 1]),
        row_index,
        result_column_name,
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

    result_row = pd.concat([result_row, result_prob], axis=1)
    data = pd.concat([data, result_row], axis=0)

pprint.pprint(data)
print(data.isnull().values.sum())
time_end = time.time()
print(time_end - parse_start)
for i in range(1, predict_rank + 1):
    data.drop("result_0_" + str(i), axis=1, inplace=True)

pprint.pprint(data)
print(data.isnull().values.sum())
data.to_csv("./data2/" + target_year + ".csv")
