from uma_predict.db.database import Reflected, Base,engine
import uuid
from sqlalchemy.orm import Mapped,mapped_column,relationship
from datetime import date

class Race(Reflected,Base):
    __tablename__="jvd_ra"
    
class Horse(Reflected,Base):
    __tablename__="jvd_um"

class Career(Reflected,Base):
    __tablename__="jvd_se"
    
class Track(Base):
    __tablename__="track"
    id:Mapped[uuid.UUID]=mapped_column("id",primary_key=True,default=uuid.uuid4)
    keibajo_code:Mapped[str]#東京競馬場なら"05"
    start_year:Mapped[str]
    end_year:Mapped[str]
    kyori:Mapped[str]
    track_code:Mapped[str]#平地　芝　外回り　など
    babajotai_code:Mapped[str]
    count:Mapped[int]
    mean:Mapped[float]
    
    
    
Reflected.prepare(engine)