import os

from dotenv import load_dotenv
from sentinelhub import SHConfig

from plastic_detection_service.dt_util import get_past_date, get_today_str

load_dotenv(override=True)

SH_INSTANCE_ID = os.environ["SH_INSTANCE_ID"]
SH_CLIENT_ID = os.environ["SH_CLIENT_ID"]
SH_CLIENT_SECRET = os.environ["SH_CLIENT_SECRET"]

DB_USER = os.environ["DB_USER"]
DB_PW = os.environ["DB_PW"]
DB_NAME = os.environ["DB_NAME"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]

POSTGIS_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
CONTENT_TYPE = "application/octet-stream"
TIME_INTERVAL = (get_past_date(7), get_today_str())
AOI = (
    120.82481750015815,
    14.619576605802296,
    120.82562856620629,
    14.66462165084734,
)  # manilla bay
MAX_CC = 0.1

SH_CONFIG = SHConfig(
    instance_id=SH_INSTANCE_ID,
    sh_client_id=SH_CLIENT_ID,
    sh_client_secret=SH_CLIENT_SECRET,
)
