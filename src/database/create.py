import os

import psycopg2
from sqlalchemy import inspect
from sqlalchemy_utils import create_database


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
