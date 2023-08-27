from models import Track, Career, Race
from database import SessionLocal, engine
from sqlalchemy.future import select

jra_central_track_code = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]
track_code = [
    "10",
    "11",
    "12",
    "17",
    "18",
    "20",
    "21",
    "23",
    "24",
]

kyori=[
    "1000",
    "2200",
    "3400",
    "1500",
    "3600",
    "1300",
    "2500",
    "1800",
    "3000",
    "1150",
    "3200",
    "2400",
    "2000",
    "2300",
    "2100",
    "1600",
    "1200",
    "1900",
    "1400",
    "2600",
    "1700",
]

condition_code = ["0", "1", "2", "3", "4"]
db = SessionLocal()
Track.metadata.tables["track"].create(engine)

for jra_code in jra_central_track_code:
    for track in track_code:
        for condition in condition_code:
            races = db.scalars(select(Race).filter())
