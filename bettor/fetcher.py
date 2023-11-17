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
    Tansho,
    Umaren,
    Umatan,
    Wide,
    Sanrenpuku,
    Sanrentan,
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

from uma_predict.df.utils import (
    field_mapper,
    condition_mapper,
    grade_mapper,
    futan_juryo_standard,
    bataiju_standard,
    kyoso_joken_mapper,
)
from uma_predict.db.database import SessionLocal
import torch
import os


nest_asyncio.apply()


@contextlib.asynccontextmanager
async def open_browser(playwright, headless=True):
    browser = await playwright.chromium.launch(headless=headless)
    try:
        yield browser
    finally:
        await browser.close()


@contextlib.asynccontextmanager
async def open_context(browser):
    context = await browser.new_context()
    try:
        yield context
    finally:
        await context.close()


@contextlib.asynccontextmanager
async def playwright_context(headless=True):
    async with async_playwright() as playwright:
        async with open_browser(playwright, headless) as browser:
            async with open_context(browser) as context:
                yield browser, context


@contextlib.asynccontextmanager
async def open_page(headless=True):
    async with playwright_context(headless) as (browser, context):
        page = await context.new_page()
        yield page


def float_or_nan(str_value):
    try:
        return float(str_value)
    except:
        return np.nan


class Fetcher:
    def __init__(
        self,
        field_condition: int,
        kaisai_nen: str | None = None,
        kaisai_tsukihi: str | None = None,
        keibajo_code: str | None = None,
        race_bango: str | None = None,
        race_name_abbre: str | None = None,
        hist_number: int = 5,
        db=SessionLocal(),
    ) -> None:
        self.kaisai_nen = kaisai_nen
        self.kaisai_tsukihi = kaisai_tsukihi
        self.keibajo_code = keibajo_code
        self.race_bango = race_bango
        self.race_name_abbre = race_name_abbre  # 1回中山5日など
        self.field_condition = field_condition
        self.path = f"./data/{self.kaisai_nen}/{self.kaisai_tsukihi}/{self.keibajo_code}/{self.race_bango}"
        self.shusso_tosu = None
        self.toroku_tosu = None
        self.tansho_odds = None
        self.fukusho_odds_low = None
        self.fukusho_odds_up = None
        self.wide_odds_low = None
        self.wide_odds_up = None
        self.umaren_odds = None
        self.umatan_odds = None
        self.sanrenpuku_odds = None
        self.sanrentan_odds = None
        self.horse_data = None
        self.odds_df = None
        self.db = db
        self.final_df_columns = []
        self.final_df_columns += initial_column
        self.hist_number = hist_number
        for i in range(1, 19):
            current_horse_column = [
                str(i) + column for column in current_horse_column_master
            ]
            self.final_df_columns += current_horse_column
            for j in range(1, hist_number + 1):
                hist_column = [
                    str(i) + column + str(j)
                    for column in horse_hist_column_master
                ]
                self.final_df_columns += hist_column

    def setup_dir(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def get_odds_from_db(self):
        data = self.db.scalars(
            select(Tansho).filter(
                Tansho.kaisai_nen == self.kaisai_nen,
                Tansho.kaisai_tsukihi == self.kaisai_tsukihi,
                Tansho.keibajo_code == self.keibajo_code,
                Tansho.race_bango == self.race_bango,
            )
        ).one()
        self.toroku_tosu = int(data.toroku_tosu)
        tansho = data.odds_tansho.rstrip()
        tansho_str = [tansho[i : i + 8] for i in range(0, len(tansho), 8)]
        tansho_odds = np.zeros(self.toroku_tosu)
        for tansho in tansho_str:
            tansho_odds[int(tansho[0:2]) - 1] = float_or_nan(tansho[2:6]) * 0.1
        self.tansho_odds = tansho_odds

        fukusho = data.odds_fukusho.rstrip()
        fukusho_str = [fukusho[i : i + 12] for i in range(0, len(fukusho), 12)]
        fukusho_odds_low = np.zeros(self.toroku_tosu)
        fukusho_odds_up = np.zeros(self.toroku_tosu)
        for fukusho in fukusho_str:
            fukusho_odds_low[int(fukusho[0:2]) - 1] = (
                float_or_nan(fukusho[2:6]) * 0.1
            )
            fukusho_odds_up[int(fukusho[0:2]) - 1] = (
                float_or_nan(fukusho[6:10]) * 0.1
            )
        self.fukusho_odds_low = fukusho_odds_low
        self.fukusho_odds_up = fukusho_odds_up

        data = self.db.scalars(
            select(Umaren).filter(
                Umaren.kaisai_nen == self.kaisai_nen,
                Umaren.kaisai_tsukihi == self.kaisai_tsukihi,
                Umaren.keibajo_code == self.keibajo_code,
                Umaren.race_bango == self.race_bango,
            )
        ).one()
        striped = data.odds_umaren.replace(" ", "")
        odds_str = [striped[i : i + 13] for i in range(0, len(striped), 13)]
        target_odds = np.zeros((self.toroku_tosu, self.toroku_tosu))
        for odd in odds_str:
            target_odds[int(odd[0:2]) - 1, int(odd[2:4]) - 1] = (
                float_or_nan(odd[4:10]) * 0.1
            )
        self.umaren_odds = target_odds

        data = self.db.scalars(
            select(Umatan).filter(
                Umatan.kaisai_nen == self.kaisai_nen,
                Umatan.kaisai_tsukihi == self.kaisai_tsukihi,
                Umatan.keibajo_code == self.keibajo_code,
                Umatan.race_bango == self.race_bango,
            )
        ).one()

        striped = data.odds_umatan.replace(" ", "")
        odds_str = [striped[i : i + 13] for i in range(0, len(striped), 13)]
        target_odds = np.zeros((self.toroku_tosu, self.toroku_tosu))
        for odd in odds_str:
            target_odds[int(odd[0:2]) - 1, int(odd[2:4]) - 1] = (
                float_or_nan(odd[4:10]) * 0.1
            )
        self.umatan_odds = target_odds

        data = self.db.scalars(
            select(Wide).filter(
                Wide.kaisai_nen == self.kaisai_nen,
                Wide.kaisai_tsukihi == self.kaisai_tsukihi,
                Wide.keibajo_code == self.keibajo_code,
                Wide.race_bango == self.race_bango,
            )
        ).one()
        striped = data.odds_wide.replace(" ", "")
        odds_str = [striped[i : i + 17] for i in range(0, len(striped), 17)]
        target_odds_low = np.zeros((self.toroku_tosu, self.toroku_tosu))
        target_odds_up = np.zeros((self.toroku_tosu, self.toroku_tosu))
        for odd in odds_str:
            target_odds_low[int(odd[0:2]) - 1, int(odd[2:4]) - 1] = (
                float_or_nan(odd[4:9]) * 0.1
            )
            target_odds_up[int(odd[0:2]) - 1, int(odd[2:4]) - 1] = (
                float_or_nan(odd[9:14]) * 0.1
            )
        self.wide_odds_low = target_odds_low
        self.wide_odds_up = target_odds_up

        data = self.db.scalars(
            select(Sanrentan).filter(
                Sanrentan.kaisai_nen == self.kaisai_nen,
                Sanrentan.kaisai_tsukihi == self.kaisai_tsukihi,
                Sanrentan.keibajo_code == self.keibajo_code,
                Sanrentan.race_bango == self.race_bango,
            )
        ).one()
        striped = data.odds_sanrentan.replace(" ", "")
        odds_str = [striped[i : i + 17] for i in range(0, len(striped), 17)]
        target_odds = np.zeros(
            (self.toroku_tosu, self.toroku_tosu, self.toroku_tosu)
        )
        for odd in odds_str:
            target_odds[
                int(odd[0:2]) - 1, int(odd[2:4]) - 1, int(odd[4:6]) - 1
            ] = (float_or_nan(odd[6:13]) * 0.1)
        self.sanrentan_odds = target_odds

        data = self.db.scalars(
            select(Sanrenpuku).filter(
                Sanrenpuku.kaisai_nen == self.kaisai_nen,
                Sanrenpuku.kaisai_tsukihi == self.kaisai_tsukihi,
                Sanrenpuku.keibajo_code == self.keibajo_code,
                Sanrenpuku.race_bango == self.race_bango,
            )
        ).one()
        striped = data.odds_sanrenpuku.replace(" ", "")
        odds_str = [striped[i : i + 15] for i in range(0, len(striped), 15)]
        target_odds = np.zeros(
            (self.toroku_tosu, self.toroku_tosu, self.toroku_tosu)
        )
        for odd in odds_str:
            target_odds[
                int(odd[0:2]) - 1, int(odd[2:4]) - 1, int(odd[4:6]) - 1
            ] = (float_or_nan(odd[6:12]) * 0.1)
        self.sanrenpuku_odds = target_odds

    def get_horse_data_from_db(self, hist_number: int):
        db = self.db
        race = db.scalars(
            select(Race).filter(
                Race.kaisai_nen == self.kaisai_nen,
                Race.kaisai_tsukihi == self.kaisai_tsukihi,
                Race.keibajo_code == self.keibajo_code,
                Race.race_bango == self.race_bango,
            )
        ).one()
        race_df = shutsubahyo(hist_number, race, db)
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
        )
        result_df = race_df[["result"]].T.copy()
        result_df.columns = ["result" + str(i) for i in range(1, 19)]
        race_df.drop(columns="result", inplace=True)
        for i in range(0, 18):
            result_horse_columns = [
                str(i) + column for column in race_df.columns
            ]
            horse_row = race_df.iloc[[i]].copy()
            horse_row.index = [0]
            horse_row.columns = result_horse_columns
            result_row = pd.concat([result_row, horse_row], axis=1)
        result_df.index = [0]
        result_row = pd.concat([result_row, result_df], axis=1)
        np_result_row = result_row.to_numpy()
        result_tensor = torch.from_numpy(np_result_row).float()
        self.horse_data = result_tensor

    def setup_odds_df_db(self):
        self.get_odds_from_db()
        large_num = 1000000

        tansho_odds = np.nan_to_num(self.tansho_odds, nan=large_num)
        fukusho_odds_low = np.nan_to_num(
            self.fukusho_odds_low, nan=large_num
        )
        fukusho_odds_up = np.nan_to_num(
            self.fukusho_odds_up, nan=large_num
        )
        wide_odds_low = np.nan_to_num(self.wide_odds_low, nan=large_num)
        wide_odds_up = np.nan_to_num(self.wide_odds_up, nan=large_num)
        umaren_odds = np.nan_to_num(self.umaren_odds, nan=large_num)
        umatan_odds = np.nan_to_num(self.umatan_odds, nan=large_num)
        umatan_odds = np.where(
            umatan_odds == 0, large_num, umatan_odds
        )
        wide_low_nagashi = (
            1
            / (
                wide_odds_low
                + wide_odds_low.T
                + np.eye(self.toroku_tosu) * large_num
            )
        ).sum(axis=1)
        wide_up_nagashi = (
            1
            / (
                wide_odds_up
                + wide_odds_up.T
                + np.eye(self.toroku_tosu) * large_num
            )
        ).sum(axis=1)
        umaren_nagashi = (
            1
            / (
                umaren_odds
                + umaren_odds.T
                + np.eye(self.toroku_tosu) * large_num
            )
        ).sum(axis=1)
        umatan_first_nagashi = (
            1 / (umatan_odds + np.eye(self.toroku_tosu) * large_num)
        ).sum(axis=1)

        umatan_second_nagashi = (
            1 / (umatan_odds + np.eye(self.toroku_tosu) * large_num)
        ).sum(axis=0)

        sanrenpuku_odds = np.nan_to_num(
            self.sanrenpuku_odds, nan=large_num
        )
        sanrenpuku_sime = (
            sanrenpuku_odds
            + np.transpose(sanrenpuku_odds, (0, 2, 1))
            + np.transpose(sanrenpuku_odds, (1, 0, 2))
            + np.transpose(sanrenpuku_odds, (1, 2, 0))
            + np.transpose(sanrenpuku_odds, (2, 0, 1))
            + np.transpose(sanrenpuku_odds, (2, 1, 0))
        )
        sanrenpuku_sime = np.where(
            sanrenpuku_sime == 0, large_num, sanrenpuku_sime
        )
        sanrenpuku_nagashi = (1 / sanrenpuku_sime).sum(axis=(1, 2))

        sanrentan_odds = np.nan_to_num(
            self.sanrentan_odds, nan=large_num
        )
        sanrentan = np.where(
            sanrentan_odds == 0, large_num, sanrentan_odds
        )
        sanrentan_inv = 1 / sanrentan
        sanrentan_first_nagashi = sanrentan_inv.sum(axis=(1, 2))
        sanrentan_second_nagashi = sanrentan_inv.sum(axis=(0, 2))
        sanrentan_third_nagashi = sanrentan_inv.sum(axis=(0, 1))

        odds_df = pd.DataFrame(
            {
                "tansho": 1 / tansho_odds,
                "fukusho_low": 1 / fukusho_odds_low,
                "fukusho_high": 1 / fukusho_odds_up,
                "wide_nagashi_low": wide_low_nagashi,
                "wide_nagashi_high": wide_up_nagashi,
                "umaren_nagashi": umaren_nagashi,
                "umatan_first_nagashi": umatan_first_nagashi,
                "umatan_second_nagashi": umatan_second_nagashi,
                "sanrenpuku_nagashi": sanrenpuku_nagashi,
                "sanrentan_first_nagashi": sanrentan_first_nagashi,
                "sanrentan_second_nagashi": sanrentan_second_nagashi,
                "sanrentan_third_nagashi": sanrentan_third_nagashi,
            }
        )
        if self.toroku_tosu != 18:
            zero_df = pd.DataFrame(
                0.0,
                index=list(range(int(self.toroku_tosu), 18)),
                columns=odds_df.columns,
            )
            odds_df = pd.concat([odds_df, zero_df], axis=0)
        row_index=0
        one_race = pd.DataFrame(
            {
                "number_of_horse": int(self.toroku_tosu) / 18.0,
            },
            index=[row_index],
        )
        permu_list = list(range(0, 18))
        swapped_race_df = odds_df.reindex(index=permu_list)

        
        for i in range(0, 18):
            result_horse_columns = [
                str(i) + column for column in swapped_race_df.columns
            ]
            result_horse_df = swapped_race_df.iloc[[i]].copy()
            result_horse_df.index = [row_index]
            result_horse_df.columns = result_horse_columns
            one_race = pd.concat([one_race, result_horse_df], axis=1)

        self.odds_df=one_race


    def get_recent_odds_from_jra(self):
        return asyncio.run(self.__get_recent_odds_from_jra())

    async def __get_recent_odds_from_jra(self):
        async with playwright_context() as (browser, context):
            page = await context.new_page()
            page.set_default_timeout(10000)
            await Fetcher.go_recent_race_odds(
                int(self.race_bango), page, self.race_name_abbre
            )
            async with page.expect_navigation():
                await page.get_by_role("link", name="人気順").click()
            odds_table = page.locator("#odds_list table")
            shusso_tosu = await odds_table.locator("tbody tr").count()
            self.shusso_tosu = shusso_tosu
            tansho_odds = np.zeros(shusso_tosu)
            fukusho_odds_low = np.zeros(shusso_tosu)
            fukusho_odds_up = np.zeros(shusso_tosu)
            for tr in await odds_table.locator("tbody tr").all():
                number = int(await tr.locator("td.num").inner_text())
                tansho_value = float_or_nan(
                    await tr.locator("td.odds_tan").inner_text()
                )
                fukusho_value_low = float_or_nan(
                    await tr.locator("td.odds_fuku span.min").inner_text()
                )
                fukusho_value_up = float_or_nan(
                    await tr.locator("td.odds_fuku span.max").inner_text()
                )
                tansho_odds[number - 1] = tansho_value
                fukusho_odds_low[number - 1] = fukusho_value_low
                fukusho_odds_up[number - 1] = fukusho_value_up

            self.tansho_odds = tansho_odds
            self.fukusho_odds_low = fukusho_odds_low
            self.fukusho_odds_up = fukusho_odds_up

            await page.get_by_role("link", name="馬連").click()
            umaren_odds = await self.scraip_odds_two_horse(page, shusso_tosu)
            self.umaren_odds = umaren_odds

            await page.get_by_role("link", name="ワイド").click()
            wide_odds_min = np.zeros((shusso_tosu, shusso_tosu))
            wide_odds_max = np.zeros((shusso_tosu, shusso_tosu))
            odds_tables = page.locator("#odds_list ul")
            for li in await odds_tables.locator("li").all():
                for tr in await li.locator("tbody tr").all():
                    umaban_str = (
                        await tr.locator("td.num").inner_text()
                    ).split("-")

                    umaban = [int(umaban_str[0]), int(umaban_str[1])]
                    wide_value_min = float_or_nan(
                        await tr.locator("td.odds span.min").inner_text()
                    )
                    wide_value_max = float_or_nan(
                        await tr.locator("td.odds span.max").inner_text()
                    )
                    wide_odds_min[
                        umaban[0] - 1, umaban[1] - 1
                    ] = wide_value_min
                    wide_odds_max[
                        umaban[0] - 1, umaban[1] - 1
                    ] = wide_value_max

            self.wide_odds_low = wide_odds_min
            self.wide_odds_up = wide_odds_max

            await page.get_by_role("link", name="馬単").click()
            umatan_odds = await self.scraip_odds_two_horse(page, shusso_tosu)
            self.umatan_odds = umatan_odds

            await page.get_by_role("link", name="3連単").click()
            odds_tables = page.locator("#contents").locator("#odds_list")
            sanrentan_odds = np.zeros((shusso_tosu, shusso_tosu, shusso_tosu))
            for ul in await odds_tables.locator("ul").all():
                for li in await ul.locator("li").all():
                    for tr in await li.locator("tbody tr").all():
                        umaban_str = (
                            await tr.locator("td.num").inner_text()
                        ).split("-")

                        umaban = [
                            int(umaban_str[0]),
                            int(umaban_str[1]),
                            int(umaban_str[2]),
                        ]
                        sanrentan_value = float_or_nan(
                            await tr.locator("td.odds").inner_text()
                        )

                        sanrentan_odds[
                            umaban[0] - 1, umaban[1] - 1, umaban[2] - 1
                        ] = sanrentan_value
            self.sanrentan_odds = sanrentan_odds

            await page.get_by_role("link", name="3連複").click()
            odds_tables = page.locator("#odds_list ul")
            sanrenpuku_odds = np.zeros((shusso_tosu, shusso_tosu, shusso_tosu))
            for li in await odds_tables.locator("li").all():
                for tr in await li.locator("tbody tr").all():
                    umaban_str = (
                        await tr.locator("td.num").inner_text()
                    ).split("-")

                    umaban = [
                        int(umaban_str[0]),
                        int(umaban_str[1]),
                        int(umaban_str[2]),
                    ]
                    sanrenpuku_value = float_or_nan(
                        await tr.locator("td.odds").inner_text()
                    )
                    sanrenpuku_odds[
                        umaban[0] - 1, umaban[1] - 1, umaban[2] - 1
                    ] = sanrenpuku_value
            self.sanrenpuku_odds = sanrenpuku_odds

    def read_horse_data_from_csv(self):
        df = pd.read_csv(f"{self.path}/horsedata.csv", index_col=0)
        np_data = df.to_numpy()
        self.horse_data = torch.from_numpy(np_data).float()

    def get_horse_data_from_jra(self):
        return asyncio.run(self.__get_horse_data_from_jra())

    async def __get_horse_data_from_jra(self):
        data = pd.DataFrame(0.0, index=[0], columns=self.final_df_columns)
        async with playwright_context() as (browser, context):
            page = await context.new_page()
            await Fetcher.go_recent_race_odds(
                int(self.race_bango), page, self.race_name_abbre
            )
            async with page.expect_navigation():
                await page.get_by_role("link", name="人気順").click()
            race_title = page.locator(
                "div.race_header div.left div.race_title"
            )
            race_grade_raw = race_title.locator(
                "div.inner div.txt h2 span.grade_icon.lg img"
            )
            grade = 0
            if await race_grade_raw.count() > 0:
                grade_alt = await race_grade_raw.get_attribute("alt")
                grade = (
                    0
                    if race_grade_raw is None
                    else 6
                    if grade_alt == "GⅠ"
                    else 5
                    if grade_alt == "GⅡ"
                    else 4
                    if grade_alt == "GⅢ"
                    else 3
                    if grade_alt == "リステッド"
                    else 0
                )
            print(f"grade:{grade}")
            race_info = race_title.locator("div.txt div.type")

            race_course = race_info.locator("div.cell.course")
            kyori_raw = await race_course.inner_text()

            kyori = float_or_nan(kyori_raw.split("：")[1][:5].replace(",", ""))

            print(f"kyori:{kyori}")
            field_str = await race_course.locator("span.detail").inner_text()
            field = 1 if field_str[1] == "芝" else 0
            target_kaisai_tsukihi = datetime.datetime.strptime(
                self.kaisai_nen + self.kaisai_tsukihi, "%Y%m%d"
            )
            odds_table = page.locator("#odds_list table")
            shusso_tosu = await odds_table.locator("tbody tr").count()

            race_condition_dict = {
                "field": field,
                "field_condition": self.field_condition,
                "grade": grade,
                "distance": (kyori - 1000.0) / 2600.0,
                "number_of_horse": shusso_tosu / 18,
            }

            data.loc[
                0, race_condition_dict.keys()
            ] = race_condition_dict.values()
            self.toroku_tosu = await odds_table.locator("tbody tr").count()

            for tr in await odds_table.locator("tbody tr").all():
                umaban = await tr.locator("td.num").inner_text()
                futan_juryo_row = float_or_nan(
                    await tr.locator("td.weight").inner_text()
                )
                print(await tr.locator("td.num").inner_text())
                bataiju_raw = float_or_nan(
                    (await tr.locator("td.h_weight").inner_text())[:3]
                )
                print(f"bataiju_raw:{bataiju_raw}")
                tansho_odds_raw = float_or_nan(
                    await tr.locator("td.odds_tan").inner_text()
                )
                tansho_ninki_raw = float_or_nan(
                    await tr.locator("td.pop").inner_text()
                )

                async with page.expect_navigation():
                    await tr.locator("td.horse").get_by_role("link").click()
                horse_page = page.locator("div.main")
                horse_profile = horse_page.locator("div.data")
                seibetu_raw = (
                    await horse_profile.locator("li.data_col2")
                    .nth(0)
                    .locator("dl dd")
                    .inner_text()
                )
                seibetu = (
                    1
                    if seibetu_raw == "牡"
                    else 2
                    if seibetu_raw == "せん"
                    else 0
                )
                seinengappi_raw = (
                    await horse_profile.locator("li.data_col2")
                    .nth(2)
                    .locator("dl dd")
                    .inner_text()
                )
                seinengappi = datetime.datetime.strptime(
                    seinengappi_raw, "%Y年%m月%d日"
                )

                current_horse_dict = {
                    "barei": (target_kaisai_tsukihi - seinengappi).days
                    / (8 * 365),
                    "seibetu": seibetu,
                    "bataiju": bataiju_standard(bataiju_raw),
                    "futan_juryo": futan_juryo_standard(futan_juryo_row * 10),
                    "tansho_odds": 1 / tansho_odds_raw,
                    "tansho_ninkijun": 1 / tansho_ninki_raw,
                }
                current_horse_dict_keys = [
                    umaban + column for column in current_horse_dict.keys()
                ]
                data.loc[
                    0, current_horse_dict_keys
                ] = current_horse_dict.values()

                hist = await horse_page.locator(
                    "ul.unit_list.mt15 div.race_detail tbody tr"
                ).all()
                hist_count = 0
                for column in hist:
                    race_link = column.locator("td").nth(2).get_by_role("link")
                    chakujun = float_or_nan(
                        await column.locator("td").nth(7).inner_text()
                    )
                    ninki = await column.locator("td").nth(6).inner_text()
                    if await race_link.count() == 0 or not ninki:
                        print("continue")
                        continue
                    else:
                        date_row = await column.locator("td.date").inner_text()
                        hist_date = datetime.datetime.strptime(
                            date_row, "%Y年%m月%d日"
                        )
                        kyori_and_field = (
                            await column.locator("td").nth(3).inner_text()
                        )
                        hist_kyori_raw = float_or_nan(kyori_and_field[1:])

                        hist_field = 1 if kyori_and_field[0] == "芝" else 0

                        hist_field_condition_raw = (
                            await column.locator("td").nth(4).inner_text()
                        )
                        hist_field_condition = (
                            1
                            if hist_field_condition_raw == "良"
                            else 2
                            if hist_field_condition_raw == "稍重"
                            else 3
                            if hist_field_condition_raw == "重"
                            else 4
                            if hist_field_condition_raw == "不良"
                            else 0
                        )
                        ninkijun_raw = float_or_nan(
                            await column.locator("td").nth(6).inner_text()
                        )
                        kinryo_raw = float_or_nan(
                            await column.locator("td").nth(9).inner_text()
                        )
                        hist_bataiju_raw = float_or_nan(
                            await column.locator("td").nth(10).inner_text()
                        )

                        async with page.expect_navigation():
                            await race_link.click()
                        result = page.locator(
                            "#contents #race_result .race_result_unit table.basic.narrow-xy.striped"
                        )
                        race_title = result.locator(
                            "caption div.race_header div.left div.race_title"
                        )
                        hist_race_grade_raw = race_title.locator(
                            "h2 span.grade_icon.lg img"
                        )
                        hist_grade = 0
                        if await hist_race_grade_raw.count() > 0:
                            hist_race_grade_alt = (
                                await hist_race_grade_raw.get_attribute("alt")
                            )
                            hist_grade = (
                                6
                                if hist_race_grade_alt == "GⅠ"
                                else 5
                                if hist_race_grade_alt == "GⅡ"
                                else 4
                                if hist_race_grade_alt == "GⅢ"
                                else 3
                                if hist_race_grade_alt == "リステッド"
                                else 0
                            )
                        race_info = race_title.locator("div.type")

                        result_table = result.locator("tbody tr").all()
                        time_list = []
                        last_3f_list = []
                        first_horse_time = 0
                        target_horse_umaban = 0
                        target_horse_time = 0.0
                        target_horse_last_3f = 0.0
                        corner_3 = -1
                        corner_4 = -1
                        hist_tansho_ninki = -1
                        hist_tansho_odds = -1
                        for tr in await result_table:
                            past_chakujun = float_or_nan(
                                await tr.locator("td.place").inner_text()
                            )
                            if np.isnan(past_chakujun):
                                continue
                            else:
                                time_raw = await tr.locator(
                                    "td.time"
                                ).inner_text()
                                hist_soha_time = float(
                                    time_raw[0]
                                ) * 60 + float(time_raw[2:])
                                time_list.append(hist_soha_time)
                                hist_last_3f = float_or_nan(
                                    await tr.locator("td.f_time").inner_text()
                                )
                                last_3f_list.append(hist_last_3f)
                                if past_chakujun == 1.0:
                                    first_horse_time = hist_soha_time
                                if past_chakujun == chakujun:
                                    target_horse_time = hist_soha_time
                                    target_horse_last_3f = hist_last_3f

                                    corner_3_raw = (
                                        await tr.locator("td.corner")
                                        .get_by_title("3コーナー通過順位")
                                        .count()
                                    )

                                    if corner_3_raw > 0:
                                        corner_3 = float_or_nan(
                                            await tr.locator("td.corner")
                                            .get_by_title("3コーナー通過順位")
                                            .inner_text()
                                        )
                                    else:
                                        corner_3 = -1
                                    corner_4_raw = (
                                        await tr.locator("td.corner")
                                        .get_by_title("4コーナー通過順位")
                                        .count()
                                    )

                                    if corner_4_raw:
                                        corner_4 = float_or_nan(
                                            await tr.locator("td.corner")
                                            .get_by_title("4コーナー通過順位")
                                            .inner_text()
                                        )
                                    else:
                                        corner_4 = -1
                                    target_horse_umaban = int(
                                        await tr.locator("td.num").inner_text()
                                    )
                                    hist_tansho_ninki = int(
                                        await tr.locator("td.pop").inner_text()
                                    )

                                    async with page.expect_navigation():
                                        await result.locator(
                                            "caption div.race_header div.right"
                                        ).get_by_role(
                                            "link", name="オッズ"
                                        ).click()
                                    hist_tansho_odds = float_or_nan(
                                        await page.locator(
                                            "div#contents div#contentsBody div#odds_list tbody tr"
                                        )
                                        .nth(target_horse_umaban - 1)
                                        .locator("td.odds_tan")
                                        .inner_text()
                                    )
                                    async with page.expect_navigation():
                                        await page.go_back()

                        time_average = statistics.mean(time_list)
                        time_std = statistics.pstdev(time_list)
                        last_3f_average = statistics.mean(last_3f_list)
                        last_3f_std = statistics.pstdev(last_3f_list)
                        hist_dict = {
                            "barei": (hist_date - seinengappi).days
                            / (8 * 365),
                            "bataiju": bataiju_standard(hist_bataiju_raw),
                            "futan_juryo": futan_juryo_standard(
                                kinryo_raw * 10
                            ),
                            "condition": hist_field_condition,
                            "field": hist_field,
                            "grade_code": hist_grade,
                            "kyori": (hist_kyori_raw - 1000.0) / 2600.0,
                            "time_sa": 4
                            - (target_horse_time - first_horse_time),
                            "soha_time_relative": 4
                            - ((target_horse_time - time_average) / time_std),
                            "kohan_3f_relative": 4
                            - (
                                (target_horse_last_3f - last_3f_average)
                                / last_3f_std
                            ),
                            "nyusen_juni": 1 / chakujun,
                            "corner_3": 1 / corner_3,
                            "corner_4": 1 / corner_4,
                            "tansho_odds": 1 / hist_tansho_odds,
                            "tansho_ninkijun": 1 / hist_tansho_ninki,
                        }
                        hist_dict_keys = [
                            umaban + column + str(hist_count + 1)
                            for column in hist_dict.keys()
                        ]
                        data.loc[0, hist_dict_keys] = hist_dict.values()
                        async with page.expect_navigation():
                            await page.go_back()
                        hist_count += 1
                    if hist_count == self.hist_number:
                        break
                async with page.expect_navigation():
                    await page.go_back()

            print(data)
            # data.to_csv(f"{self.path}/horsedata.csv")
            np_data = data.to_numpy()
            tensor_data = torch.from_numpy(np_data).float()
            self.horse_data = tensor_data

    @staticmethod
    async def scraip_odds_two_horse(page, shusso_tosu):
        umaren_odds = np.zeros((shusso_tosu, shusso_tosu))
        odds_tables = page.locator("#odds_list ul")
        for li in await odds_tables.locator("li").all():
            for tr in await li.locator("tbody tr").all():
                umaban_str = (await tr.locator("td.num").inner_text()).split(
                    "-"
                )
                umaban = [int(umaban_str[0]), int(umaban_str[1])]
                umaren_value = float_or_nan(
                    await tr.locator("td.odds").inner_text()
                )
                umaren_odds[umaban[0] - 1, umaban[1] - 1] = umaren_value

        return umaren_odds

    @staticmethod
    async def go_recent_race_odds(race_number: int, page, race_name_abbre):
        await page.goto("https://jra.jp/")
        async with page.expect_navigation():
            await page.get_by_role("link", name="オッズ").click()
        async with page.expect_navigation():
            await page.get_by_role("link", name=race_name_abbre).click()
        async with page.expect_navigation():
            await page.locator("body").locator("#contents").locator(
                "#contentsBody"
            ).locator(".race_select").locator("table").locator("tbody tr").nth(
                race_number - 1
            ).locator(
                "th.race_num"
            ).get_by_role(
                "link"
            ).click()
