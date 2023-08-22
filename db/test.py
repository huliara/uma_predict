from models import Race
from database import SessionLocal
from sqlalchemy.future import select

db = SessionLocal()

races = db.scalars(select(Race))

