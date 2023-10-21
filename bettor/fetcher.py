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

# asyncioの入れ子を許す
import nest_asyncio
from uma_predict.df.alpha_df import shutsubahyo, race_condition, hist_number
from uma_predict.df.utils import field_mapper, condition_mapper, grade_mapper
from db.database import SessionLocal
import torch

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


async def select_1st_race(page, place: str):  # placeは"1回中山5日"など
    await page.goto("https://jra.jp/")
    async with page.expect_navigation():
        await page.get_by_role("link", name="レース結果").click()
    async with page.expect_navigation():
        await page.get_by_role("link", name=place).click()
    async with page.expect_navigation():
        await page.get_by_role("link", name="1レースオッズ", exact=True).click()


async def get_race_count(page):
    return await page.locator("ul.race-num").first.locator("li").count()


async def select_race(page, race):
    async with page.expect_navigation():
        await page.get_by_role("link", name=f"{race}レース").first.click()


def float_or_nan(str_value):
    try:
        return float(str_value)
    except:
        return np.nan


# 単勝オッズ取得
async def get_odds_tansho(page, place, race):
    number_list = []
    odds_list = []
    nagashi_list = []
    umatan_table = await get_odds_umatan(page, place, race)

    async with page.expect_navigation():
        await page.get_by_role("link", name="単勝・複勝").click()
    odds_table = page.locator("#odds_list table")
    for tr in await odds_table.locator("tbody tr").all():
        number = int(await tr.locator("td.num").inner_text())
        odds = float_or_nan(await tr.locator("td.odds_tan").inner_text())
        nagashi = umatan_table[umatan_table["number1"] == number]["odds"]
        print(f"{race}レース")
        print(nagashi)
        nagashi_value = combine_odds(nagashi)
        number_list.append(number)
        odds_list.append(odds)
        nagashi_list.append(nagashi_value)
    print(nagashi_list)
    table = pd.DataFrame(
        {"number": number_list, "odds": odds_list, "nagashi": nagashi_list}
    )
    table["place"] = place
    table["race"] = race

    # 列の並び替え
    columns = ["place", "race", "number", "odds", "nagashi"]
    table = table[columns]

    return table


# 馬連オッズ取得
async def get_odds_umaren(page, place, race):
    number1_list = []
    number2_list = []
    odds_list = []
    nagashi_list = []
    umatan_table = await get_odds_umatan(page, place, race)

    async with page.expect_navigation():
        await page.get_by_role("link", name="馬連").click()
    odds_tables = page.locator("#odds_list table.umaren")
    for odds_table in await odds_tables.all():
        number1 = int(await odds_table.locator("caption").inner_text())
        for tr in await odds_table.locator("tbody tr").all():
            number2 = int(await tr.locator("th").inner_text())
            odds = float_or_nan(await tr.locator("td").inner_text())
            number1_list.append(number1)
            number2_list.append(number2)
            odds_list.append(odds)
            nagashi = umatan_table.query(
                "(number1==@number1 & number2==@number2)|(number1==@number2 & number2==@number1)"
            )["odds"]
            nagashi_value = combine_odds(nagashi)
            nagashi_list.append(nagashi_value)

    table = pd.DataFrame(
        {
            "number1": number1_list,
            "number2": number2_list,
            "odds": odds_list,
            "nagashi": nagashi_list,
        }
    )
    table["place"] = place
    table["race"] = race

    # 列の並び替え
    columns = ["place", "race", "number1", "number2", "odds", "nagashi"]
    table = table[columns]

    return table


# 馬単オッズ取得
async def get_odds_umatan(page, place, race):
    number1_list = []
    number2_list = []
    odds_list = []

    async with page.expect_navigation():
        await page.get_by_role("link", name="馬単").click()
    odds_tables = page.locator("#odds_list table.umatan")
    for odds_table in await odds_tables.all():
        number1 = int(await odds_table.locator("caption").inner_text())
        for tr in await odds_table.locator("tbody tr").all():
            number2 = int(await tr.locator("th").inner_text())
            odds = float_or_nan(await tr.locator("td").inner_text())
            number1_list.append(number1)
            number2_list.append(number2)
            odds_list.append(odds)

    table = pd.DataFrame(
        {
            "number1": number1_list,
            "number2": number2_list,
            "odds": odds_list,
        }
    )
    table["place"] = place
    table["race"] = race

    # 列の並び替え
    columns = ["place", "race", "number1", "number2", "odds"]
    table = table[columns]

    return table


# 開催のオッズを取得する
async def get_odds(place):
    odds_tansho_list = []
    odds_umaren_list = []
    odds_umatan_list = []
    async with playwright_context() as (browser, context):
        page = await context.new_page()
        await select_1st_race(page, place)

        race_count = await get_race_count(page)
        for race in range(1, race_count + 1):
            await select_race(page, race)
            odds_tansho = await get_odds_tansho(page, place, race)
            odds_umaren = await get_odds_umaren(page, place, race)
            odds_umatan = await get_odds_umatan(page, place, race)
            odds_tansho_list.append(odds_tansho)
            odds_umaren_list.append(odds_umaren)
            odds_umatan_list.append(odds_umatan)
    odds_tansho = pd.concat(odds_tansho_list, ignore_index=True)
    odds_umaren = pd.concat(odds_umaren_list, ignore_index=True)
    odds_umatan = pd.concat(odds_umatan_list, ignore_index=True)
    return odds_tansho, odds_umaren, odds_umatan


# オッズを取得して保存
def get_odds_and_save_csv(place, save_dir, csv_base, version):
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    tansho_path = save_dir / f"{csv_base}_tansho_{version}.csv"
    umaren_path = save_dir / f"{csv_base}_umaren_{version}.csv"
    umatan_path = save_dir / f"{csv_base}_umatan_{version}.csv"
    if tansho_path.exists() or umaren_path.exists():
        print("Already exists.")
        return

    try:
        odds_tansho, odds_umaren, odds_umatan = asyncio.run(get_odds(place))
        odds_tansho.to_csv(tansho_path, index=False)
        odds_umaren.to_csv(umaren_path, index=False)
        odds_umatan.to_csv(umatan_path, index=False)
    except:
        traceback.print_exc()
        print("Failed to get odds.")


get_odds_and_save_csv("3回東京3日", "odds_csv", "20230430_tokyo_3", "0930")


class Fetcher:
    def __init__(
        self,
        kaisai_nen: str,
        kaisai_tsukihi: str,
        keibajo_code: str,
        race_bago: str,
        race_name_abbre: str,
        db=SessionLocal(),
    ) -> None:
        self.kaisai_nen = kaisai_nen
        self.kaisai_tsukihi = kaisai_tsukihi
        self.keibajo_code = keibajo_code
        self.race_bango = race_bago
        self.race_name_abbre = race_name_abbre  # 1回中山5日など
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
        self.db = db

    def get_odds_from_db(self):
        data = self.db.scalars(
            select(Tansho).filter(
                Tansho.kaisai_nen == self.kaisai_nen,
                Tansho.kaisai_tsukihi == self.kaisai_tsukihi,
                Tansho.keibajo_code == self.keibajo_code,
                Tansho.race_bango == self.race_bango,
            )
        ).one()
        tansho = data.odds_tansho.rstrip()
        tansho_str = [tansho[i : i + 8] for i in range(0, len(tansho), 8)]
        tansho_odds = np.empty(len(tansho_str))
        for tansho in tansho_str:
            tansho_odds[int(tansho[0:2]) - 1] = float_or_nan(tansho[2:6]) * 0.1
        self.tansho_odds = tansho_odds

        fukusho = data.odds_fukusho.rstrip()
        fukusho_str = [fukusho[i : i + 12] for i in range(0, len(fukusho), 12)]
        fukusho_odds_low = np.empty(len(fukusho_str))
        fukusho_odds_up = np.empty(len(fukusho_str))
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
        striped = data.odds_umaren.rstrip()
        odds_str = [striped[i : i + 13] for i in range(0, len(striped), 13)]
        target_odds = np.zeros((len(odds_str), len(odds_str)))
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
        striped = data.odds_umatan.rstrip()
        odds_str = [striped[i : i + 13] for i in range(0, len(striped), 13)]
        target_odds = np.empty((len(odds_str), len(odds_str)))
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
        striped = data.odds_wide.rstrip()
        odds_str = [striped[i : i + 13] for i in range(0, len(striped), 13)]
        target_odds_low = np.zeros((len(odds_str), len(odds_str)))
        target_odds_up = np.zeros((len(odds_str), len(odds_str)))
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
        striped = data.odds_sanrentan.rstrip()
        odds_str = [striped[i : i + 17] for i in range(0, len(striped), 17)]
        target_odds = np.empty((len(odds_str), len(odds_str), len(odds_str)))
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
        striped = data.odds_sanrenpuku.rstrip()
        odds_str = [striped[i : i + 15] for i in range(0, len(striped), 15)]
        target_odds = np.zeros((len(odds_str), len(odds_str), len(odds_str)))
        for odd in odds_str:
            target_odds[
                int(odd[0:2]) - 1, int(odd[2:4]) - 1, int(odd[4:6]) - 1
            ] = (float_or_nan(odd[6:12]) * 0.1)
        self.sanrenpuku_odds = target_odds

    def get_horse_data_from_db(self):
        db = self.db
        race = db.scalars(
            select(Race).filter(
                Race.kaisai_nen == self.kaisai_nen,
                Race.kaisai_tsukihi == self.kaisai_tsukihi,
                Race.keibajo_code == self.keibajo_code,
                Race.race_bango == self.race_bango,
            )
        ).one()
        race_df = shutsubahyo(race, db)
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
        )
        result_df = race_df[["result"]].T.copy()
        result_df.columns = ["result" + str(i) for i in range(1, 19)]
        race_df.drop(columns="result", inplace=True)
        for i in range(0, 18):
            result_horse_columns = [
                str(i) + column for column in race_df.columns
            ]
            horse_row = race_df.iloc[[i]].copy()
            horse_row.columns = result_horse_columns
            result_row = pd.concat([result_row, horse_row], axis=1)
        result_row = pd.concat([result_row, result_df], axis=1)
        np_result_row = result_row.to_numpy()
        result_tensor = torch.from_numpy(np_result_row).float()

        self.horse_data = result_tensor

    async def get_recent_odds_from_jra(self):
        async with playwright_context() as (browser, context):
            page = await context.new_page()
            await Fetcher.go_recent_race_odds(
                int(self.race_bango), page, self.race_name_abbre
            )
            async with page.expect_navigation():
                await page.get_by_role("link", name="人気順").click()
            odds_table = page.locator("#odds_list table")
            shusso_tosu = await odds_table.locator("tbody tr").count()
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
            for li in odds_tables.locator("li").all():
                for tr in li.locator("tbody tr").all():
                    umaban_str = (
                        await tr.locator("td.num").inner_text().split("-")
                    )
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

            await page.get_by_role("link", name="三連単").click()
            odds_tables = page.locator("#odds_list")
            sanrentan_odds = np.zeros((shusso_tosu, shusso_tosu, shusso_tosu))
            for ul in odds_table.locator("ul").all():
                for li in ul.locator("li").all():
                    for tr in li.locator("tbody tr").all():
                        umaban_str = (
                            await tr.locator("td.num").inner_text().split("-")
                        )
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

            await page.get_by_role("link", name="三連複").click()
            odds_tables = page.locator("#odds_list ul")
            sanrenpuku_odds = np.zeros((shusso_tosu, shusso_tosu, shusso_tosu))
            for li in odds_tables.locator("li").all():
                for tr in li.locator("tbody tr").all():
                    umaban_str = (
                        await tr.locator("td.num").inner_text().split("-")
                    )
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


    #deprecated
    async def get_horse_data_from_jra(self):
        async with playwright_context() as (browser, context):
            page = await context.new_page()
            await Fetcher.go_recent_race_odds(
                int(self.race_bango), page, self.race_name_abbre
            )
            race_title = page.locator(
                "div.race_header div.left div.race_title"
            )
            race_grade = race_title.locator("h2 span.grade_icon.lg")
            race_info = race_title.locator("div.type")
            race_class = race_info.locator("div.cell.class").inner_text()
            race_course = race_info.locator("div.cell.course")
            kyori = race_course.inner_text()
            field = race_course.locator("span.detail").inner_text()

            odds_table = page.locator("#odds_list table")
            shusso_tosu = await odds_table.locator("tbody tr").count()
            shutsubahyo = np.zeros(shusso_tosu, shusso_tosu)
            for tr in await odds_table.locator("tbody tr").all():
                number = int(await tr.locator("td.num").inner_text())

                futan_juryo = float_or_nan(
                    await tr.locator("td.weight").inner_text()
                )
                horse_name = await tr.locator("td.horse").inner_text()
                async with page.expect_navigation():
                    await tr.locator("td.horse").get_by_role("link").click()
                horse_page = page.locator("div.main")
                horse_profile = horse_page.locator("div.data")
                seibetu = (
                    await horse_profile.locator("li.data_col2")
                    .nth(0)
                    .locator("dl dd")
                    .inner_text()
                )
                seinengappi = (
                    await horse_profile.locator("li.data_col2")
                    .nth(2)
                    .locator("dl dd")
                    .inner_text()
                )

                hist = await horse_page.locator(
                    "ul.unit_list.mt15 tbody tr"
                ).all()
                hist_list = []
                hist_count = 0
                for column in hist:
                    race_link = (
                        await column.locator("td").nth(2).get_by_role("link")
                    )
                    chakujun = float_or_nan(
                        await column.locator("td").nth(7).inner_text()
                    )
                    if race_link and chakujun:
                        date = await column.locator("td.date").inner_text()
                        place = await column.locator("td").nth(1).inner_text()
                        kyori_and_field = (
                            await column.locator("td").nth(3).inner_text()
                        )
                        field_condition = (
                            await column.locator("td").nth(4).inner_text()
                        )
                        ninkijun = (
                            await column.locator("td").nth(6).inner_text()
                        )
                        kinryo = await column.locator("td").nth(9).inner_text()
                        bataiju = (
                            await column.locator("td").nth(10).inner_text()
                        )
                        soha_time = (
                            await column.locator("td").nth(11).inner_text()
                        )
                        async with page.expect_navigation():
                            race_link.click()
                        result = page.locator(
                            "div#contents div#race_result table.basic.narrow-xy.striped"
                        )
                        race_title = result.locator(
                            "caption div.race_header div.left div.race_title"
                        )
                        race_grade = race_title.locator(
                            "h2 span.grade_icon.lg"
                        )
                        race_info = race_title.locator("div.type")
                        race_class = race_info.locator(
                            "div.cell.class"
                        ).inner_text()
                        result_table = result.locator("tbody tr").all()
                        time_list = []
                        last_3f_list = []
                        first_horse_time = 0
                        target_horse_umaban = 0
                        for tr in result_table:
                            past_chakujun = float_or_nan(
                                await tr.locator("td").nth(7).inner_text()
                            )
                            if past_chakujun:
                                time_list.append(
                                    await tr.locator("td").nth(10).inner_text()
                                )
                                last_3f_list.append(
                                    await tr.locator("td").nth(11).inner_text()
                                )
                            if past_chakujun == 1.0:
                                first_horse_time = (
                                    await tr.locator("td").nth(10).inner_text()
                                )
                            if past_chakujun == chakujun:
                                target_horse_time = (
                                    await tr.locator("td.time")inner_text()
                                )
                                target_horse_last_3f = (
                                    await tr.locator("td.f_time").nth(11).inner_text()
                                )
                                corner_3= (
                                    await tr.locator("td.corner").get_by_title("3コーナー通過順位").inner_text()
                                )
                                corner_4= (
                                    await tr.locator("td.corner").get_by_title("4コーナー通過順位").inner_text()
                                )
                                target_horse_umaban=int(
                                    await tr.locator("td.num").inner_text()
                                )
                                tansho_ninki=(
                                    await tr.locator("td.pop").inner_text()
                                )
                                async with page.expect_navigation():
                                    await result.locator("caption div.race_header div.right").get_by_role("link",name="オッズ").click()
                                tansho_odds=float_or_nan(await page.locator("div#contents div#contentsBody div#odds_list tbody tr").nth(target_horse_umaban-1).locator("td.odds_tan").inner_text())
                                
                    else:
                        continue
                    if hist_count == hist_number:
                        break
    
    async def get_horse_data_from_netkeiba(self):


    @staticmethod
    async def scraip_odds_two_horse(page, shusso_tosu):
        umaren_odds = np.zeros((shusso_tosu, shusso_tosu))
        odds_tables = page.locator("#odds_list ul")
        for li in odds_tables.locator("li").all():
            for tr in li.locator("tbody tr").all():
                umaban_str = await tr.locator("td.num").inner_text().split("-")
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
            ).locator("#race_select").locator("table").locator("tbody tr").nth(
                race_number - 1
            ).get_by_role(
                "link"
            ).click()
