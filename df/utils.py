import datetime
from uma_predict.db.models import Race, Career, Horse, Track
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from sqlalchemy import desc
import pandas as pd
import numpy as np
import pprint


def horse_to_list(horse, seinengappi):
    horse = [
        (
            datetime.datetime.strptime(
                horse.kaisai_nen + horse.kaisai_tsukihi, "%Y%m%d"
            )
            - datetime.datetime.strptime(seinengappi, "%Y%m%d")
        ).days,
        0
        if horse.seibetsu_code == "2"
        else 1
        if horse.seibetsu_code == "1"
        else 2,
        int(horse.bataiju),
        int(horse.blinker_shiyo_kubun),
        float(horse.futan_juryo),
        float(horse.tansho_odds),
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


def parse_time_sa(time_sa: str):
    return float(time_sa[1:]) * 0.1 if time_sa[0] == "+" else 0.0


def race_nyusen_result(race: Race, db: Session):
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
    return horses


def horse_history(
    horse: Horse,
    db: Session,
    start: str = "20020615",
    end: datetime = "20230830",
) -> list[tuple[Career, Race]]:
    career_race_hist = []
    horse_hist = db.scalars(
        select(Career)
        .filter(
            Career.ketto_toroku_bango == horse.ketto_toroku_bango,
            Career.kaisai_nen + Career.kaisai_tsukihi < end,
            Career.kaisai_nen + Career.kaisai_tsukihi >= start,
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
    for hist in horse_hist:
        race = db.scalars(
            select(Race).filter(
                Race.kaisai_nen == hist.kaisai_nen,
                Race.kaisai_tsukihi == hist.kaisai_tsukihi,
                Race.keibajo_code == hist.keibajo_code,
                Race.race_bango == hist.race_bango,
            )
        ).one()
        career_race_hist.append((hist, race))
    return career_race_hist


def hist_model_to_pandas(
    career: tuple[Career, Race], columns: list[str], index: int
):
    race_dict = career[0].__dict__
    race_dict |= career[1].__dict__
    data = pd.DataFrame(race_dict, index=[index])
    filterd = data[columns]
    
    return filterd

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
    bataiju_mean = 469.12418300653593
    bataiju_std = 28.81961143804636
    return 2 + (float(bataiju) - bataiju_mean) / bataiju_std


def futan_juryo_standard(futan_juryo: str):
    return (float(futan_juryo) - 510.0) / 90.0
