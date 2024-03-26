import os

import psycopg2
from sqlalchemy import create_engine, inspect
from sqlalchemy_utils import create_database, database_exists

from plastic_detection_service.config import DATABASE_URL
from plastic_detection_service.database.models import Base


def create_postgis_db(engine):
    create_database(url=engine.url)
    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PW"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION postgis")
    conn.commit()
    cursor.close()
    conn.close()


def check_tables_exists(engine):
    ins = inspect(engine)
    for _t in ins.get_table_names():
        print(_t)


def create_tables(engine, base):
    base.metadata.drop_all(engine)
    base.metadata.create_all(engine)

    check_tables_exists(engine)
    engine.dispose()


def create_triggers():
    custom_trigger_function_sql = """
    CREATE OR REPLACE FUNCTION prevent_duplicate_image_insert()
    RETURNS TRIGGER AS $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM images
            WHERE timestamp = NEW.timestamp
            AND ST_Equals(bbox, NEW.bbox)
            AND image_id = NEW.image_id
        ) THEN
            RAISE EXCEPTION 'Image with the same timestamp, bbox and sentinel_hub_id already exists';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    CREATE TRIGGER prevent_duplicate_image_insert
    BEFORE INSERT ON images
    FOR EACH ROW
    EXECUTE FUNCTION prevent_duplicate_image_insert();
    """

    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PW"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )
    cursor = conn.cursor()
    cursor.execute(custom_trigger_function_sql)
    cursor.execute(trigger_sql)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
