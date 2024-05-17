import datetime

from shapely.geometry import Polygon

from src import config
from src.database.connect import create_db_session
from src.database.insert import Insert

insert = Insert(session=create_db_session())
model = insert.insert_model(
    model_id=config.RUNDPOD_MODEL_ID, model_url=config.RUNPOD_ENDPOINT_ID
)

geom = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "coordinates": [
            [
                [120.53947145910576, 14.79964156981643],
                [120.53947145910576, 14.438016273402596],
                [120.9926404870738, 14.438016273402596],
                [120.9926404870738, 14.79964156981643],
                [120.53947145910576, 14.79964156981643],
            ]
        ],
        "type": "Polygon",
    },
}

aoi = insert.insert_aoi(
    name="manilla bay",
    created_at=datetime.datetime.now(),
    geometry=Polygon(geom["geometry"]["coordinates"][0]),
)

job = insert.insert_job(aoi_id=aoi.id, model_id=1)
