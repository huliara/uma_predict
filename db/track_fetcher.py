from track import track_mean
from database import SessionLocal,engine
from models import Track
import pprint
kaisyu_period = {
    "01": [("2006", "2012"), ("2014", "2022")],
    "03": [("2006", "2010"), ("2012", "2022")],
    "07": [("2006", "2010"), ("2012", "2022")],
    "08": [("2006", "2020")],
}
db = SessionLocal()
mean=track_mean("2006","2012","01",db)
for mea in mean:
    pprint.pprint(mea.__dict__)
db.add_all(mean)
db.commit()
print(len(mean))