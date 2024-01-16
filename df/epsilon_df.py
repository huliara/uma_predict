import pandas as pd
from uma_predict.db.models import Race
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from uma_predict.df.utils import (
    field_mapper,
    grade_mapper,
    condition_mapper,
    kyoso_joken_mapper,
)
from uma_predict.df.epsilon_hyo import (
    current_horse_column_master,
    horse_hist_column_master,
    hist_number,
    repeat_number,
    initial_column,
    shutsubahyo,
)
import numpy as np
import time
import polars as pl

parse_start = time.time()

np.random.seed(0)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)
db = SessionLocal()


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
    print(race.kaisai_nen + race.kaisai_tsukihi)
    race_df = shutsubahyo(race, db)

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
                    kyoso_joken_mapper(race.kyoso_joken_code),
                    (int(race.kyori) - 1000.0) / 2600.0,
                    int(race.shusso_tosu) / 18.0,
                ]
            ],
            columns=initial_column,
            index=[row_index],
        )
        permu_list = np.random.permutation(list(range(1, 19))).tolist()
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
final_df_columns += initial_column

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
print(len(final_df_columns))
print(len(data[0]))
print(len(data))
df_schema = {i: pl.Float32 for i in final_df_columns}
final_df = pl.DataFrame(data, schema=df_schema)
print(final_df.null_count().sum(axis=1))

for column in final_df.columns:
    hadureti = final_df.select(column).filter(pl.col(column) > 20).shape[0]
    if hadureti > 0:
        print(f"{column}:{hadureti}")


time_end = time.time()
print(time_end - parse_start)
target_year = "2016-2021"
final_df.write_parquet(
    f"dataEt/{target_year}-{hist_number}-{repeat_number}.parquet"
)

data = []
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
    race_start = time.time()
    print(race.kaisai_nen + race.kaisai_tsukihi)
    race_df = shutsubahyo(race, db)

    row_index = 0

    result_row = pd.DataFrame(
        [
            [
                field_mapper(race.track_code),
                condition_mapper(race),
                grade_mapper(race.grade_code),
                kyoso_joken_mapper(race.kyoso_joken_code),
                (int(race.kyori) - 1000.0) / 2600.0,
                int(race.shusso_tosu) / 18.0,
            ]
        ],
        columns=initial_column,
        index=[row_index],
    )
    permu_list = list(range(1, 19))

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
final_df_columns += initial_column

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
final_df.write_parquet(f"dataEt/{target_year}-{hist_number}-1.parquet")
