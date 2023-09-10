from track import track_mean
from database import SessionLocal
from models import Race, Career
from sqlalchemy.future import select
import matplotlib.pyplot as plt

db=SessionLocal()

keibajo_code="06"
kaisai_num=[]



for i in range(2006,2023):
    races = db.scalars(
        select(Race).filter(
            Race.kaisai_nen == str(i),
            Race.keibajo_code == keibajo_code,
            Race.track_code <= "26",
            Race.track_code >= "00",
        )
    ).all()
    kaisai_num.append(len(races))

print(kaisai_num)

plt.bar(list(range(2006,2023)),kaisai_num   )
plt.show()