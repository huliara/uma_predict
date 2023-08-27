from uma_predict.db.models import Race, Career
from uma_predict.db.database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import pprint
db = SessionLocal()

races = db.scalars(
    select(Race.kyoso_joken_code).filter(
        Race.kaisai_nen>="2003",
        Race.keibajo_code>="01",
        Race.keibajo_code<="10",
        Race.track_code <= "26",
        Race.track_code >= "00",
    )
).all()

pprint.pprint(set(races))
