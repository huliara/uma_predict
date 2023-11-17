from models import Track
from database import SessionLocal, engine
from sqlalchemy.future import select
import pprint
db = SessionLocal()

track=db.scalars(select(Track.mean).filter(Track.is_last3f==False,Track.mean>=230).order_by()).all()

for tr in track:
    pprint.pprint(tr)