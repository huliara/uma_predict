from uma_predict.db.models import Race, Career
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import pprint
import statistics

db = SessionLocal()

races = db.scalars(
    select(Race.kyori).filter(
        Race.kaisai_nen>="2006",
        Race.keibajo_code>="01",
        Race.keibajo_code<="10",
    )
).all()

bataiju_list=[int(race) for race in races]
print(set(bataiju_list))
print(statistics.stdev(bataiju_list))
