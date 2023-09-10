from models import Track, Career, Race
from database import SessionLocal, engine, Session
from sqlalchemy.future import select
import statistics
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

kyori_list = [
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


def track_mean(start_year: str, end_year: str, keibajo_code: str, db: Session):
    keibajo_datas = []
    for track in track_code:
        for condition in condition_code:
            for kyori in kyori_list:
                if track == "23" or track == "24":
                    races = db.scalars(
                        select(Race).filter(
                            Race.kaisai_nen >= start_year,
                            Race.kaisai_nen <= end_year,
                            Race.keibajo_code == keibajo_code,
                            Race.track_code == track,
                            Race.kyori == kyori,
                            Race.babajotai_code_dirt == condition,
                        )
                    )
                else:
                    races = db.scalars(
                        select(Race).filter(
                            Race.kaisai_nen >= start_year,
                            Race.kaisai_nen <= end_year,
                            Race.keibajo_code == keibajo_code,
                            Race.track_code == track,
                            Race.kyori == kyori,
                            Race.babajotai_code_shiba == condition,
                        )
                    )

                soha_time_sum = 0.0
                last3f_times_sum = 0.0
                soha_horse_num = 0
                for race in races:
                    careers: list[Career] = db.scalars(
                        select(Career).filter(
                            Career.kaisai_nen == race.kaisai_nen,
                            Career.kaisai_tsukihi == race.kaisai_tsukihi,
                            Career.keibajo_code == race.keibajo_code,
                            Career.race_bango == race.race_bango,
                            Career.keibajo_code >= "01",
                            Career.ijo_kubun_code != "1",
                            Career.ijo_kubun_code != "2",
                            Career.ijo_kubun_code != "3",
                            Career.ijo_kubun_code != "4",
                            Career.nyusen_juni != "00",
                        )
                    ).all()

                    soha_times = [
                        float(career.soha_time[0]) * 60
                        + float(career.soha_time[1:]) * 0.1
                        for career in careers
                    ]
                    last3f_times = [
                        float(career.kohan_3f) * 0.1 for career in careers
                    ]
                    horse_num = len(careers)
                    if horse_num > 0:
                        soha_time_sum += sum(soha_times)
                        last3f_times_sum += sum(last3f_times)
                        soha_horse_num += horse_num
                if soha_horse_num == 0:
                    continue
                mean = soha_time_sum / soha_horse_num
                std=statistics.stdev(soha_times)
                mean_last3f = last3f_times_sum / soha_horse_num
                std_last3f=statistics.stdev(last3f_times)
                track_data = Track(
                    keibajo_code=keibajo_code,
                    start_year=start_year,
                    end_year=end_year,
                    kyori=kyori,
                    track_code=track,
                    babajotai_code=condition,
                    count=soha_horse_num,
                    mean=mean,
                    std=std,
                )
                track_data_last3f = Track(
                    keibajo_code=keibajo_code,
                    start_year=start_year,
                    end_year=end_year,
                    kyori=kyori,
                    track_code=track,
                    babajotai_code=condition,
                    count=soha_horse_num,
                    mean=mean_last3f,
                    std=std_last3f,
                    is_last3f=True,
                )
                keibajo_datas.append(track_data)
                keibajo_datas.append(track_data_last3f)
    return keibajo_datas
