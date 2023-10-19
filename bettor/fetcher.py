import asyncio
import contextlib
from playwright.async_api import Playwright, async_playwright, expect
import numpy as np
import pandas as pd

import asyncio
from pathlib import Path
import traceback

# asyncioの入れ子を許す
import nest_asyncio

from db.database import SessionLocal


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
        self.race_bago = race_bago
        self.race_name_abbre = race_name_abbre  # 1回中山5日など
        self.tansho_odds = None
        self.fukusho_odds = None
        self.wide_odds = None
        self.umaren_odds = None
        self.umatan_odds = None
        self.sanrenpuku_odds = None
        self.sanrentan_odds = None
        self.horse_data = None
        self.db = db

    def get_odds_from_db(self):
        pass

    async def get_recent_odds_from_jra(self):
        async with playwright_context() as (browser, context):
            page = await context.new_page()
            await Fetcher.go_recent_race_odds(
                page,self.race_name_abbre
            )
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

            

    def get_horse_data_from_jra(self):
        pass

    @staticmethod
    async def go_recent_race_odds(self, page, race_name_abbre):
        await page.goto("https://jra.jp/")
        async with page.expect_navigation():
            await page.get_by_role("link", name="オッズ").click()
        async with page.expect_navigation():
            await page.get_by_role("link", name=race_name_abbre).click()
        async with page.expect_navigation():
            