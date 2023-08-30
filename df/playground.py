from uma_predict.db.models import Race, Career
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import pprint
db = SessionLocal()

races = db.scalars(
    select(Career).filter(
        Career.kaisai_nen>="2006",
        Career.nyusen_juni=="00",
        Career.keibajo_code>="01",
        Career.keibajo_code<="10",
        Career.ijo_kubun_code=="0",
        Career.bataiju!="   ",
    )
).all()

for race in races:
    pprint.pprint(race.__dict__)

print(len(races))