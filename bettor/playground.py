from uma_predict.db.models import Umatan,Tansho
from uma_predict.bettor.fetcher import Fetcher
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
import pprint
import numpy as np

db = SessionLocal()

umatan = db.scalars(select(Tansho).filter(Tansho.kaisai_nen >= '2014')).all()


for umatan in umatan:
    pprint.pprint(umatan.odds_tans)
'''

def float_or_nan(str_value):
    try:
        return float(str_value)
    except:
        return np.nan

print(float_or_nan('----'))
'''