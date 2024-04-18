from src import config
from src.database.connect import create_db_session
from src.database.insert import Insert

insert = Insert(session=create_db_session())

insert.insert_model(config.RUNDPOD_MODEL_ID, config.RUNPOD_ENDPOINT_ID)
