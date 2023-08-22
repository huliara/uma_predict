from models import Track, Career
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
    "00",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "27",
    "28",
    "29",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
]

condition_code = [
    "0",
    "1",
    "2",
    "3",
    "4"
]
db = SessionLocal()
Track.metadata.tables["track"].create(engine)

for jra_code in jra_central_track_code:
    for track in track_code:
        for condition in condition_code:
            db.