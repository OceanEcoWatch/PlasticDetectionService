from sqlalchemy.orm import Session

from plastic_detection_service.database.db import (
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
