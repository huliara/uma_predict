from models import Race, Career, Horse,Track
from database import SessionLocal
from sqlalchemy.future import select
from sqlalchemy.orm import Session
import pprint

db = SessionLocal()


races = db.scalars(
    select(Race).filter(
        Race.kaisai_nen >= "2003",
        Race.keibajo_code >= "01",
        Race.keibajo_code <= "10",
        Race.track_code <= "26",
        Race.track_code >= "00",
    )
).first()

pprint.pprint(races.__dict__)

career = db.scalars(
    select(Career).filter(
        Career.kaisai_nen == races.kaisai_nen,
        Career.keibajo_code == races.keibajo_code,
        Career.kaisai_tsukihi == races.kaisai_tsukihi,
        Career.race_bango == races.race_bango,
        Career.ijo_kubun_code == "0",
    )
).all()

track=db.scalars(
    select(Track).filter(
        Track.keibajo_code=="05"
    )
).all()

uma = db.scalars(select(Horse).filter(Horse.seinengappi >= "2019")).first()
pprint.pprint(races.__dict__)