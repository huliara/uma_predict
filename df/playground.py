from uma_predict.db.models import Race, Career
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import pprint
db = SessionLocal()

races = db.scalars(
    select(Career).filter(
        Career.kaisai_nen=="2006",
        Career.keibajo_code=="08",
        Career.kaisai_tsukihi=="0105",
        Career.race_bango=="11",
        Career.umaban=="10",
    )
).all()

for race in races:
    pprint.pprint(race.__dict__)
