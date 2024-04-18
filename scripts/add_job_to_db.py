import datetime

from shapely.geometry import Polygon

from src import config
from src.database.connect import create_db_session
from src.database.insert import Insert

insert = Insert(session=create_db_session())
model = insert.insert_model(
    model_id=config.RUNDPOD_MODEL_ID, model_url=config.RUNPOD_ENDPOINT_ID
)
aoi = insert.insert_aoi(
    name="manilla bay",
    created_at=datetime.datetime.now(),
    geometry=Polygon(
        [
            (120.82481750015815, 14.619576605802296),
            (120.82562856620629, 14.619576605802296),
            (120.82562856620629, 14.66462165084734),
            (120.82481750015815, 14.66462165084734),
            (120.82481750015815, 14.619576605802296),
        ]
    ),
)

job = insert.insert_job(aoi_id=aoi.id, model_id=1)
