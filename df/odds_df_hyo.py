
import numpy as np
import pandas as pd
from sqlalchemy.future import select
from uma_predict.db.models import (
    Race,
    Career,
)

from uma_predict.bettor.fetcher import Fetcher
from uma_predict.df.utils import (

    rank_probability_v2,
    chakusa_num_v2,
)
from uma_predict.db.database import SessionLocal
import time
np.set_printoptions(linewidth=1400)

start_time = time.time()
db = SessionLocal()

initial_column = [
    "number_of_horse",
]


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

def odds_df(race:Race):
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
    return odds_df


