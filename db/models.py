from database import Reflected, Base,engine

class Race(Reflected,Base):
    __tablename__="jvd_ra"
    
class Horse(Reflected,Base):
    __tablename__="jvd_um"

class Career(Reflected,Base):
    __tablename__="jvd_se"


Reflected.prepare(engine)
