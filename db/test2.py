from track import track_mean
from database import SessionLocal
from models import Race, Career
from sqlalchemy.future import select
import matplotlib.pyplot as plt
import numpy as np
import pprint
def soha_time_parser(soha_time: str):
    return float(soha_time[0]) * 60 + float(soha_time[1:]) * 0.1
db = SessionLocal()

keibajo_code = "06"
kaisai_num = []
means = np.zeros((10, 17), dtype=np.float64)

keibajo_code_list=["01","02","03","04","05","06","07","08","09","10"]

for j in range(1, 11):
    for i in range(2006, 2023):
        races_time = 0
        races_count = 0
        races = db.scalars(
            select(Race).filter(
                Race.kaisai_nen == str(i),
                Race.keibajo_code == keibajo_code_list[j-1],
                Race.kyoso_joken_code != "701",
                Race.kyori=="2000",
                Race.track_code <= "22",
                Race.track_code >= "00",
                Race.nyusen_tosu > "03",
            )
        ).all()
        if len(races)==0:
            continue
        for race in races:
            results = db.scalars(
                select(Career).filter(
                    Career.kaisai_nen == race.kaisai_nen,
                    Career.kaisai_tsukihi == race.kaisai_tsukihi,
                    Career.keibajo_code == race.keibajo_code,
                    Career.race_bango == race.race_bango,
                )
            ).all()
            races_count += len(results)
            for result in results:
                races_time += soha_time_parser(result.soha_time)
                
        means[j-1,i-2006]=races_time/races_count

pprint.pprint(means)

for i in range(10):
    plt.plot(range(1990, 2023), means[i], label=str(i+1))
plt.ylim(115, 130)
plt.show()
