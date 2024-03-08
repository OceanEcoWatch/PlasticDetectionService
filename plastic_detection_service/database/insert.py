from sqlalchemy.orm import Session

from plastic_detection_service.database.db import (
    Image,
    PredictionVector,
    SceneClassificationVector,
    get_db_engine,
)


class Insert:
    def __init__(self, session: Session, engine=get_db_engine()):
        self.session = session
        self.engine = engine

    def insert_image(self, image: Image):
        self.session.add(image)
        self.session.commit()

    def insert_prediction(self, prediction: PredictionVector):
        self.session.add(prediction)
        self.session.commit()

    def insert_scl(self, scl: SceneClassificationVector):
        self.session.add(scl)
        self.session.commit()
