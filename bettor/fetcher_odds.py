import asyncio
import contextlib
from playwright.async_api import Playwright, async_playwright, expect
import numpy as np
import pandas as pd
import asyncio
import nest_asyncio
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
    ) -> None:
        self.kaisai_nen = kaisai_nen
        self.kaisai_tsukihi = kaisai_tsukihi
        self.keibajo_code = keibajo_code
        self.race_bango = race_bango
        self.race_name_abbre = race_name_abbre  # 1回中山5日など
        self.field_condition = field_condition
        self.path = f"./data/{self.kaisai_nen}/{self.kaisai_tsukihi}/{self.keibajo_code}/{self.race_bango}"
        self.shusso_tosu = None
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
        self.final_df_columns = []
    def setup_dir(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    
    
    def setup_odds_df_db(self):
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
                + np.eye(self.shusso_tosu) * large_num
            )
        ).sum(axis=1)
        wide_up_nagashi = (
            1
            / (
                wide_odds_up
                + wide_odds_up.T
                + np.eye(self.shusso_tosu) * large_num
            )
        ).sum(axis=1)
        umaren_nagashi = (
            1
            / (
                umaren_odds
                + umaren_odds.T
                + np.eye(self.shusso_tosu) * large_num
            )
        ).sum(axis=1)
        umatan_first_nagashi = (
            1 / (umatan_odds + np.eye(self.shusso_tosu) * large_num)
        ).sum(axis=1)

        umatan_second_nagashi = (
            1 / (umatan_odds + np.eye(self.shusso_tosu) * large_num)
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
        if self.shusso_tosu != 18:
            zero_df = pd.DataFrame(
                0.0,
                index=list(range(int(self.shusso_tosu), 18)),
                columns=odds_df.columns,
            )
            odds_df = pd.concat([odds_df, zero_df], axis=0)
        row_index=0
        one_race = pd.DataFrame(
            {
                "number_of_horse": int(self.shusso_tosu) / 18.0,
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
