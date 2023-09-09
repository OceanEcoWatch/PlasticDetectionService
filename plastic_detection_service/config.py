import os

from dotenv import load_dotenv
from sentinelhub import SHConfig

load_dotenv()

config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials")

DB_USER = os.environ["DB_USER"]
DB_PW = os.environ["DB_PW"]
DB_NAME = os.environ["DB_NAME"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]

POSTGIS_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
