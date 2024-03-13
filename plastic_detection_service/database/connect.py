from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from plastic_detection_service.config import DATABASE_URL
from plastic_detection_service.database.models import (
    Image,
    PredictionVector,
    SceneClassificationVector,
    get_db_engine,
)
from plastic_detection_service.models import DownloadResponse, Raster, Vector


class Insert:
    def __init__(self, session: Session, engine=get_db_engine()):
        self.session = session
        self.engine = engine

    def insert_image(
        self, raster: Raster, download_response: DownloadResponse, image_url: str
    ):
        with self.session as session:
            image = Image.from_response_and_raster(download_response, raster, image_url)
            session.add(image)
            session.commit()

    def insert_prediction(self, prediction: Vector, image_id: int, model_id: int):
        with self.session as session:
            pv = PredictionVector.from_vector(prediction, image_id, model_id)
            session.add(pv)
            session.commit()

    def insert_scl(self, scl: Vector, image_id: int):
        with self.session as session:
            sv = SceneClassificationVector.from_vector(scl, image_id)
            session.add(sv)
            session.commit()


class DatabaseError(Exception):
    def __init__(self, message):
        super().__init__(message)


def create_db_session():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


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
