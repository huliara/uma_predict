from uma_predict.db.models import Umatan,Tansho
from uma_predict.bettor.fetcher import Fetcher
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
import pprint
import numpy as np


fetcher = Fetcher(race_bango="1",race_name_abbre="4回東京9日")
with fetcher.get_recent_odds_from_jra():
    print(fetcher.sanrentan_odds)