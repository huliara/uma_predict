import pandas as pd
from uma_predict.db.models import Race, Career, Horse, Track
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy import desc
from utils import (
    field_mapper,
    grade_mapper,
    roll_mapper,
    condition_mapper,
    soha_time_parser,
)
import datetime
import numpy as np
import time
import polars as pl
import statistics

parse_start = time.time()
bataiju_mean = 469.12418300653593
bataiju_std = 28.81961143804636
np.random.seed(0)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)
db = SessionLocal()

race_condition = [
    "field",
    "field_condition",
    "grade",
    "distance",
    "number_of_horse",
]

current_horse_column_master = [
    "barei",
    "seibetu",
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
    "condition",
    "field",
    "grade_code",
    "kyori",
    "soha_time",
    "time_sa",
    "soha_time_relative",
    "kohan_3f",
    "kohan_3f_relative",
    "nyusen_juni",
    "corner_3",
    "corner_4",
    "tansho_odds",
    "tansho_ninkijun",
]


hist_number = 5
repeat_number = 1


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


def rank_probability(chakusa_num: list, tosu_limit: int = 8):
    prob_list = np.zeros(18)
    prob_list[0] = 1
    for j in range(0, tosu_limit):
        prob_list[j + 1] = prob_list[j] * (1 - chakusa_num[j])
        prob_list[j] = prob_list[j] * chakusa_num[j]
    return prob_list


def bataiju_standard(bataiju: str):
    return 2 + (float(bataiju) - bataiju_mean) / bataiju_std


def futan_juryo_standard(futan_juryo: str):
    return (float(futan_juryo) - 510.0) / 90.0


"""カラム名を作成"""
race_df_columns = []
race_df_columns += current_horse_column_master

for i in range(1, hist_number + 1):
    hist_column = [column + str(i) for column in horse_hist_column_master]
    race_df_columns += hist_column


data = []

races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen <= "2021",
        Race.kaisai_nen >= "2016",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code <= "26",
        Race.track_code >= "00",
        Race.nyusen_tosu > "03",
        Race.kyoso_joken_code != "701",
    )
).all()

for race in races:
    race_start = time.time()
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

    """着差でスムージングした正解ラベルを計算"""
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

    result_prob = rank_probability(
        chakusa_num(chakusa_list),
        tosu_limit=min([int(race.nyusen_tosu) - 1, 8]),
    )

    race_df = pd.DataFrame(
        0.0, index=list(range(1, 19)), columns=race_df_columns
    )

    race_df["result"] = 0.0

    for i in range(0, int(race.nyusen_tosu)):
        race_df.at[nyusen_umaban_list[i], "result"] = result_prob[i]

    # 特徴量を作成
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
            ).days
            / (8 * 365),
            "seibetu": 0
            if horse.seibetsu_code == "2"
            else 1
            if horse.seibetsu_code == "1"
            else 2,
            "bataiju": bataiju_standard(horse.bataiju),
            "blinker_shiyo_kubun": int(horse.blinker_shiyo_kubun),
            "futan_juryo": futan_juryo_standard(horse.futan_juryo),
            "tansho_odds": 10 / float(horse.tansho_odds),
            "tansho_ninkijun": 1 / int(horse.tansho_ninkijun),
        }

        race_df.loc[
            int(horse.umaban), current_horse_dict.keys()
        ] = current_horse_dict.values()

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
                    Track.babajotai_code == str(past_race_condition),
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
                    Track.babajotai_code == str(past_race_condition),
                    Track.is_last3f == True,
                )
            ).first()

            other_horse_hist = db.scalars(
                select(Career).filter(
                    Career.kaisai_nen == hist.kaisai_nen,
                    Career.kaisai_tsukihi == hist.kaisai_tsukihi,
                    Career.keibajo_code == hist.keibajo_code,
                    Career.race_bango == hist.race_bango,
                    Career.ijo_kubun_code != "1",
                    Career.ijo_kubun_code != "2",
                    Career.ijo_kubun_code != "3",
                    Career.ijo_kubun_code != "4",
                    Career.nyusen_juni != "00",
                )
            ).all()

            other_horse_soha_time = [
                soha_time_parser(other_hist.soha_time)
                for other_hist in other_horse_hist
            ]

            soha_time_average = statistics.mean(other_horse_soha_time)
            soha_time_std = statistics.pstdev(other_horse_soha_time)
            other_horse_kohan_3f = [
                float(other_hist.kohan_3f) * 0.1
                for other_hist in other_horse_hist
            ]
            kohan_3f_average = statistics.mean(other_horse_kohan_3f)
            kohan_3f_std = statistics.pstdev(other_horse_kohan_3f)

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
                "bataiju": bataiju_standard(hist.bataiju),
                "blinker_shiyo_kubun": int(hist.blinker_shiyo_kubun),
                "futan_juryo": futan_juryo_standard(hist.futan_juryo),
                "condition": past_race_condition,
                "field": field_mapper(past_race.track_code),
                "grade_code": grade_mapper(past_race.grade_code),
                "kyori": (float(past_race.kyori) - 1000.0) / 2600.0,
                "soha_time": 4
                - (
                    (soha_time_parser(hist.soha_time) - time_mean.mean)
                    / time_mean.std
                ),
                "time_sa": 4 - float(hist.time_sa[1:]) * 0.1
                if hist.time_sa[0] == "+"
                else 4.0,
                "soha_time_relative": 4
                - (
                    (soha_time_parser(hist.soha_time) - soha_time_average)
                    / soha_time_std
                )
                if soha_time_std != 0
                else 0,
                "kohan_3f": 4
                - (
                    (float(hist.kohan_3f) * 0.1 - last3f_mean.mean)
                    / last3f_mean.std
                ),
                "kohan_3f_relative": 4
                - (
                    (float(hist.kohan_3f) * 0.1 - kohan_3f_average)
                    / kohan_3f_std
                ),
                "nyusen_juni": 1 / int(hist.nyusen_juni),
                "corner_3": 1 / int(hist.corner_3)
                if hist.corner_3 != "00"
                else 0,
                "corner_4": 1 / int(hist.corner_4)
                if hist.corner_4 != "00"
                else 0,
                "tansho_odds": 10 / float(hist.tansho_odds),
                "tansho_ninkijun": 1 / int(hist.tansho_ninkijun),
            }
            hist_dict_key = [
                keys + str(hist_index) for keys in hist_dict.keys()
            ]
            race_df.loc[int(horse.umaban), hist_dict_key] = hist_dict.values()

            if hist_index == hist_number:
                break
            hist_index += 1

    for p_id in range(0, repeat_number):
        row_index = int(
            race.kaisai_nen
            + race.kaisai_tsukihi
            + race.keibajo_code
            + race.race_bango
            + str(p_id)
        )

        result_row = pd.DataFrame(
            [
                [
                    field_mapper(race.track_code),
                    condition_mapper(race),
                    grade_mapper(race.grade_code),
                    (int(race.kyori) - 1000.0) / 2600.0,
                    int(race.shusso_tosu) / 18.0,
                ]
            ],
            columns=race_condition,
            index=[row_index],
        )
        permu_list = list(range(1, 19))
        #permu_list = np.random.permutation(list(range(1, 19))).tolist()
        swapped_race_df = race_df.reindex(index=permu_list)

        result_df = swapped_race_df[["result"]].T.copy()
        result_df.index = [row_index]
        result_df.columns = ["result" + str(i) for i in range(1, 19)]
        swapped_race_df.drop(columns="result", inplace=True)

        for i in range(0, 18):
            result_horse_columns = [
                str(i) + column for column in swapped_race_df.columns
            ]
            result_horse_df = swapped_race_df.iloc[[i]].copy()
            result_horse_df.index = [row_index]
            result_horse_df.columns = result_horse_columns
            result_row = pd.concat([result_row, result_horse_df], axis=1)

        result_row = pd.concat([result_row, result_df], axis=1)
        list_row = result_row.to_numpy().tolist()
        data.append(list_row[0])


final_df_columns = []
final_df_columns += race_condition

for i in range(1, 19):
    current_horse_column = [
        str(i) + column for column in current_horse_column_master
    ]
    final_df_columns += current_horse_column
    for j in range(1, hist_number + 1):
        hist_column = [
            str(i) + column + str(j) for column in horse_hist_column_master
        ]
        final_df_columns += hist_column
result_column = ["result" + str(i) for i in range(1, 19)]
final_df_columns += result_column

print(len(data[0]))
print(len(data))
final_df = pl.DataFrame(data, schema=final_df_columns)
print(final_df.null_count().sum(axis=1))

for column in final_df.columns:
    hadureti = final_df.select(column).filter(pl.col(column) > 20).shape[0]
    if hadureti > 0:
        print(f"{column}:{hadureti}")


time_end = time.time()
print(time_end - parse_start)
target_year = "2022"
final_df.write_parquet(
    f"data9/{target_year}-{hist_number}-{repeat_number}.parquet"
)
