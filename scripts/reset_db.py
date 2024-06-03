from sqlalchemy import create_engine
from sqlalchemy_utils import drop_database

from src.config import DATABASE_URL
from src.database.create import create_postgis_db, create_tables
from src.database.models import Base

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    print("Dropping database")
    drop_database(engine.url)
    print("Creating database")
    create_postgis_db(engine)
    print("Creating tables")
    create_tables(engine, Base)
