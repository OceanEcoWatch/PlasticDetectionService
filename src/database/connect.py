from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL


class DatabaseError(Exception):
    def __init__(self, message):
        super().__init__(message)


def create_db_session() -> Session:
    engine = create_engine(DATABASE_URL)
    session = sessionmaker(bind=engine)
    return session()


def _execute_query(session, query):
    result = session.execute(query)
    return result.fetchall()


def safe_execute_query(session, query):
    try:
        result = _execute_query(session, query)
        return result
    except SQLAlchemyError as e:
        session.rollback()
        error_message = f"Database error: {str(e)}"
        raise DatabaseError(error_message)


def safe_insert(session, orm_object):
    try:
        with session as s:
            s.add(orm_object)
            s.commit()
    except SQLAlchemyError as e:
        session.rollback()
        error_message = f"Database error: {str(e)}"
        raise DatabaseError(error_message)


def safe_bulk_insert(session, orm_objects):
    try:
        with session as s:
            s.bulk_save_objects(orm_objects)
            s.commit()
    except SQLAlchemyError as e:
        session.rollback()
        error_message = f"Database error: {str(e)}"
        raise DatabaseError(error_message)
