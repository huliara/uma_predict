from sqlalchemy import create_engine,types
from sqlalchemy.orm import sessionmaker, scoped_session, DeclarativeBase,Session
from sqlalchemy.ext.declarative import DeferredReflection
from uma_predict.db.env import DB_USER, DB_PASSWORD,  DB_NAME
import uuid

DATABASE = "postgresql+psycopg2://%s:%s@localhost:5432/%s" % (
    DB_USER,
    DB_PASSWORD,
    DB_NAME,)

engine = create_engine(
    DATABASE,
    echo=True
)

# 実際の DB セッション
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
)

class Base(DeclarativeBase):
    pass
Base.query = SessionLocal.query_property()

class Reflected(DeferredReflection):
    __abstract__=True



# Dependency Injection用
def get_db()->Session:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()