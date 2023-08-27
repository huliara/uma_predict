import pandas as pd
from uma_predict.db.models import Race, Career, Horse
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
import pprint
import random
pd.set_option('display.max_columns', 100)
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
history_number = 3

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

data = pd.DataFrame(columns=column_name)

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


horse_history_index = [
    histindex + umaban * 10
    for umaban in range(1, 19)
    for histindex in range(1, history_number + 1)
]

for name in data.columns:
    print()
races = db.scalars(
    select(Race)
    .filter(
        Race.kaisai_nen == "2006",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code <= "26",
        Race.track_code >= "00",
        Race.nyusen_tosu > "03",
        Race.race_bango == "11",
        Race.kyoso_joken_code != "701",
    )
    .limit(2)
).all()


def horse_to_list(horse):
    horse = [
        int(horse.umaban),
        int(horse.seibetsu_code),
        int(horse.barei),
        int(horse.bataiju),
        int(horse.blinker_shiyo_kubun),
        float(horse.futan_juryo) * 0.1,
    ]
    return horse


for race in races:
    row_index = int(
        race.kaisai_nen
        + race.kaisai_tsukihi
        + race.keibajo_code
        + race.race_bango
    )
    row = pd.DataFrame(columns=column_name, index=[row_index])
    horses = db.scalars(
        select(Career).filter(
            Career.kaisai_nen == race.kaisai_nen,
            Career.kaisai_tsukihi == race.kaisai_tsukihi,
            Career.keibajo_code == race.keibajo_code,
            Career.race_bango == race.race_bango,
            Career.ijo_kubun_code != "1",
            Career.ijo_kubun_code != "2",
            Career.ijo_kubun_code != "3",
            Career.ijo_kubun_code != "4",
        )
    ).all()
    horse_current = pd.DataFrame(
        columns=current_horse_columns, index=list(range(1, 19))
    )

    bottom_horses = list(
        filter(
            lambda x: int(x.nyusen_juni) >= int(race.nyusen_tosu) - 1, horses
        )
    )

    horses_history = pd.DataFrame(
        columns=horses_history_columns, index=[horse_history_index]
    )
    for horse in horses:
        horse_current.loc[int(horse.umaban)] = horse_to_list(horse)

        horse_hist = db.scalars(
            select(Career)
            .filter(
                Career.ketto_toroku_bango == horse.ketto_toroku_bango,
                Career.kaisai_nen + Career.kaisai_tsukihi
                <= horse.kaisai_nen + horse.kaisai_tsukihi,
                Career.ijo_kubun_code != "1",
                Career.ijo_kubun_code != "2",
                Career.ijo_kubun_code != "3",
                Career.ijo_kubun_code != "4",
            )
            .limit(history_number)
        ).all()
        if len(horse_hist) == 0:
            continue

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
                int(hist.barei),
                int(hist.umaban),
                int(hist.bataiju),
                int(hist.blinker_shiyo_kubun),
                float(hist.futan_juryo) * 0.1,
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
                int(past_race.kyori),
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

    for row in horse_current[horse_current["umaban"].isnull()].itertuples():
        bottom_horse = bottom_horses[random.randint(0, len(bottom_horses) - 1)]
        horse_current.loc[row.Index] = horse_to_list(bottom_horse)
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
    pprint.pprint(race.__dict__)
    print(horses_history)
    print(horse_current)
