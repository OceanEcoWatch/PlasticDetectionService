import os

from dotenv import load_dotenv
from sentinelhub.config import SHConfig

load_dotenv(override=True)

SH_INSTANCE_ID = os.environ["SH_INSTANCE_ID"]
SH_CLIENT_ID = os.environ["SH_CLIENT_ID"]
SH_CLIENT_SECRET = os.environ["SH_CLIENT_SECRET"]

DB_USER = os.environ["DB_USER"]
DB_PW = os.environ["DB_PW"]
DB_NAME = os.environ["DB_NAME"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNDPOD_MODEL_ID = "plastic_detection_model:1.0.1"
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]


SH_CONFIG = SHConfig(
    instance_id=SH_INSTANCE_ID,
    sh_client_id=SH_CLIENT_ID,
    sh_client_secret=SH_CLIENT_SECRET,
)

S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]

L1CBANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
