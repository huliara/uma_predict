import asyncio
import contextlib
from playwright.async_api import Playwright, async_playwright, expect
import numpy as np
import pandas as pd
from sqlalchemy.future import select
import asyncio
from pathlib import Path
import traceback
from uma_predict.db.models import (
    Horse,
    Race,
    Career,
)
import datetime
import statistics
import nest_asyncio
from uma_predict.df.epsilon_hyo import (
    shutsubahyo,
    initial_column,
    current_horse_column_master,
    horse_hist_column_master,
)
from uma_predict.bettor.fetcher import Fetcher
from uma_predict.df.utils import (
    field_mapper,
    condition_mapper,
    grade_mapper,
    kyoso_joken_mapper,
    rank_probability_v2,
    chakusa_num_v2,
)
from uma_predict.db.database import SessionLocal
import torch
import os
import polars as pl
import time
import pprint

np.set_printoptions(linewidth=1400)

start_time = time.time()
db = SessionLocal()


grade_code_key = ["A", "B", "C", "D", "E", "F", "G", "H", "L"]

kyoso_joken_key = ["005", "010", "016", "701", "703", "999"]

kyori_key = [
    "1000",
    "1150",
    "1200",
    "1300",
    "1400",
    "1500",
    "1600",
    "1700",
    "1800",
    "1900",
    "2000",
    "2100",
    "2200",
    "2300",
    "2400",
    "2500",
    "2600",
    "2750",
    "2770",
    "2850",
    "2860",
    "2880",
    "2890",
    "2910",
    "2970",
    "3000",
    "3100",
    "3110",
    "3140",
    "3170",
    "3200",
    "3210",
    "3250",
    "3290",
    "3300",
    "3330",
    "3350",
    "3380",
    "3390",
    "3400",
    "3570",
    "3600",
    "3900",
    "3930",
    "4100",
    "4250",
]

track_code_key = [
    "10",
    "11",
    "12",
    "17",
    "18",
    "20",
    "21",
    "23",
    "24",
    "52",
    "54",
    "55",
    "56",
    "57",
]

race_bango_key = [
    "race1",
    "race2",
    "race3",
    "race4",
    "race5",
    "race6",
    "race7",
    "race8",
    "race9",
    "race10",
    "race11",
    "race12",
]

keibajo_key = [
    "sapporo",
    "hakodate",
    "fukusima",
    "nigata",
    "tokyo",
    "nakayama",
    "tyuukyou",
    "kyoto",
    "hanshin",
    "kokura",
]

condition_key = [
    "ryo",
    "yayaomo",
    "omo",
    "furyo",
]

initial_column = [
    "number_of_horse",
]

race_condition_columns = (
    grade_code_key
    + kyoso_joken_key
    + kyori_key
    + track_code_key
    + race_bango_key
    + keibajo_key
    + condition_key
)

horse_odds_master = [
    "tansho",
    "fukusho_low",
    "fukusho_high",
    "wide_nagashi_low",
    "wide_nagashi_high",
    "umaren_nagashi",
    "umatan_first_nagashi",
    "umatan_second_nagashi",
    "sanrenpuku_nagashi",
    "sanrentan_first_nagashi",
    "sanrentan_second_nagashi",
    "sanrentan_third_nagashi",
]

repeat_number = 30
large_num = 1000000

races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen <= "2021",
        Race.kaisai_nen >= "2016",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code >= "10",
        Race.nyusen_tosu > "03",
    )
).all()


data = []

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
    print(race.kaisai_nen + race.kaisai_tsukihi)
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
    nyusen_umaban_list = [int(horse.umaban) - 1 for horse in horses]
    for horse in horses:
        if horse.chakusa_code_1 != "   ":
            chakusa_list[int(horse.nyusen_juni) - 2] = horse.chakusa_code_1
            if horse.chakusa_code_2 != "   ":
                chakusa_list[int(horse.nyusen_juni) - 3] = horse.chakusa_code_2
                if horse.chakusa_code_3 != "   ":
                    chakusa_list[
                        int(horse.nyusen_juni) - 4
                    ] = horse.chakusa_code_3

    result_prob = rank_probability_v2(
        chakusa_num_v2(chakusa_list),
        tosu_limit=int(race.nyusen_tosu) - 1,
    )

    fetcher.get_odds_from_db()
    fetcher.tansho_odds = np.nan_to_num(fetcher.tansho_odds, nan=large_num)
    fetcher.fukusho_odds_low = np.nan_to_num(
        fetcher.fukusho_odds_low, nan=large_num
    )
    fetcher.fukusho_odds_up = np.nan_to_num(
        fetcher.fukusho_odds_up, nan=large_num
    )
    fetcher.wide_odds_low = np.nan_to_num(fetcher.wide_odds_low, nan=large_num)
    fetcher.wide_odds_up = np.nan_to_num(fetcher.wide_odds_up, nan=large_num)
    fetcher.umaren_odds = np.nan_to_num(fetcher.umaren_odds, nan=large_num)
    fetcher.umatan_odds = np.nan_to_num(fetcher.umatan_odds, nan=large_num)
    fetcher.umatan_odds = np.where(
        fetcher.umatan_odds == 0, large_num, fetcher.umatan_odds
    )
    wide_low_nagashi = (
        1
        / (
            fetcher.wide_odds_low
            + fetcher.wide_odds_low.T
            + np.eye(fetcher.toroku_tosu) * large_num
        )
    ).sum(axis=1)
    wide_up_nagashi = (
        1
        / (
            fetcher.wide_odds_up
            + fetcher.wide_odds_up.T
            + np.eye(fetcher.toroku_tosu) * large_num
        )
    ).sum(axis=1)
    umaren_nagashi = (
        1
        / (
            fetcher.umaren_odds
            + fetcher.umaren_odds.T
            + np.eye(fetcher.toroku_tosu) * large_num
        )
    ).sum(axis=1)
    umatan_first_nagashi = (
        1 / (fetcher.umatan_odds + np.eye(fetcher.toroku_tosu) * large_num)
    ).sum(axis=1)

    umatan_second_nagashi = (
        1 / (fetcher.umatan_odds + np.eye(fetcher.toroku_tosu) * large_num)
    ).sum(axis=0)
    """
    eye_3d = np.array(
        [
            np.eye(fetcher.toroku_tosu) * large_num
            for _ in range(fetcher.toroku_tosu)
        ]
    )
    eye_3d = (
        eye_3d
        + np.transpose(eye_3d, (0, 2, 1))
        + np.transpose(eye_3d, (2, 1, 0))
    )
    """
    fetcher.sanrenpuku_odds = np.nan_to_num(
        fetcher.sanrenpuku_odds, nan=large_num
    )
    sanrenpuku_sime = (
        fetcher.sanrenpuku_odds
        + np.transpose(fetcher.sanrenpuku_odds, (0, 2, 1))
        + np.transpose(fetcher.sanrenpuku_odds, (1, 0, 2))
        + np.transpose(fetcher.sanrenpuku_odds, (1, 2, 0))
        + np.transpose(fetcher.sanrenpuku_odds, (2, 0, 1))
        + np.transpose(fetcher.sanrenpuku_odds, (2, 1, 0))
    )
    sanrenpuku_sime = np.where(
        sanrenpuku_sime == 0, large_num, sanrenpuku_sime
    )
    sanrenpuku_nagashi = (1 / sanrenpuku_sime).sum(axis=(1, 2))

    fetcher.sanrentan_odds = np.nan_to_num(
        fetcher.sanrentan_odds, nan=large_num
    )
    sanrentan = np.where(
        fetcher.sanrentan_odds == 0, large_num, fetcher.sanrentan_odds
    )
    sanrentan_inv = 1 / sanrentan
    sanrentan_first_nagashi = sanrentan_inv.sum(axis=(1, 2))
    sanrentan_second_nagashi = sanrentan_inv.sum(axis=(0, 2))
    sanrentan_third_nagashi = sanrentan_inv.sum(axis=(0, 1))

    odds_df = pd.DataFrame(
        {
            "tansho": 1 / fetcher.tansho_odds,
            "fukusho_low": 1 / fetcher.fukusho_odds_low,
            "fukusho_high": 1 / fetcher.fukusho_odds_up,
            "wide_nagashi_low": wide_low_nagashi,
            "wide_nagashi_high": wide_up_nagashi,
            "umaren_nagashi": umaren_nagashi,
            "umatan_first_nagashi": umatan_first_nagashi,
            "umatan_second_nagashi": umatan_second_nagashi,
            "sanrenpuku_nagashi": sanrenpuku_nagashi,
            "sanrentan_first_nagashi": sanrentan_first_nagashi,
            "sanrentan_second_nagashi": sanrentan_second_nagashi,
            "sanrentan_third_nagashi": sanrentan_third_nagashi,
            "result": 0.0,
        }
    )
    for i in range(0, int(race.nyusen_tosu)):
        odds_df.at[nyusen_umaban_list[i], "result"] = result_prob[i]
    if race.toroku_tosu != "18":
        zero_df = pd.DataFrame(
            0.0,
            index=list(range(int(race.toroku_tosu), 18)),
            columns=odds_df.columns,
        )
        odds_df = pd.concat([odds_df, zero_df], axis=0)
    print(odds_df)
    row_index = 0
    """
    grade_list = [0 if race.grade_code != key else 1 for key in grade_code_key]
    grade_dict = dict(zip(grade_code_key, grade_list))

    kyosso_joken_list = [
        0 if race.kyoso_joken_code != key else 1 for key in kyoso_joken_key
    ]
    kyoso_joken_dict = dict(zip(kyoso_joken_key, kyosso_joken_list))

    kyori_list = [0 if race.kyori != kyori else 1 for kyori in kyori_key]
    kyori_dict = dict(zip(kyori_key, kyori_list))

    track_code_list = [
        0 if race.track_code != key else 1 for key in track_code_key
    ]
    track_code_dict = dict(zip(track_code_key, track_code_list))

    race_bango_list = [
        0 if int(race.race_bango) != i else 1 for i in range(1, 13)
    ]
    race_bango_dict = dict(zip(race_bango_key, race_bango_list))

    keibajo_list = [
        0 if int(race.keibajo_code) != i else 1 for i in range(1, 11)
    ]
    keibajo_dict = dict(zip(keibajo_key, keibajo_list))

    condition_num = int(race.babajotai_code_dirt) + int(
        race.babajotai_code_shiba
    )
    condition_list = [0 if condition_num != i else 1 for i in range(1, 5)]
    condition_dict = dict(zip(condition_key, condition_list))

    race_condition_dict = dict(
        **grade_dict,
        **kyoso_joken_dict,
        **kyori_dict,
        **track_code_dict,
        **race_bango_dict,
        **keibajo_dict,
        **condition_dict,
    )
    race_condition_df = pd.DataFrame(race_condition_dict, index=[row_index])
    """

    for _ in range(0, repeat_number):
        one_race = pd.DataFrame(
            {
                "number_of_horse": int(race.shusso_tosu) / 18.0,
            },
            index=[row_index],
        )
        #one_race = pd.concat([one_race, race_condition_df], axis=1)
        permu_list = np.random.permutation(list(range(0, 18))).tolist()
        swapped_race_df = odds_df.reindex(index=permu_list)

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
            one_race = pd.concat([one_race, result_horse_df], axis=1)

        one_race = pd.concat([one_race, result_df], axis=1)
        list_row = one_race.to_numpy().tolist()
        data.append(list_row[0])


final_df_columns = []
final_df_columns += initial_column
#final_df_columns += race_condition_columns
for i in range(1, 19):
    current_horse_column = [str(i) + column for column in horse_odds_master]
    final_df_columns += current_horse_column
result_column = ["result" + str(i) for i in range(1, 19)]
final_df_columns += result_column
df_schema = {i: pl.Float32 for i in final_df_columns}
final_df = pl.DataFrame(data, schema=df_schema)
final_df = final_df.fill_null(0.0)
final_df = final_df.fill_nan(0.0)
print(final_df.null_count().sum(axis=1))
print(final_df.shape)
for column in final_df.columns:
    hadureti = final_df.select(column).filter(pl.col(column) > 20).shape[0]
    if hadureti > 0:
        print(f"{column}:{hadureti}")
time_end = time.time()

print(time_end - start_time)


target_year = "2016-2021"
final_df.write_parquet(f"dataOdds/{target_year}-{repeat_number}_v3.parquet")
