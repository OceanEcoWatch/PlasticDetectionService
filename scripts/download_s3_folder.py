import os

from src import config
from src.aws import s3

contents = s3.get_folder_contents(config.S3_BUCKET_NAME, "predictions")
for f in contents:
    content = s3.download_from_s3(config.S3_BUCKET_NAME, f)

    # create necessary directories
    os.makedirs(os.path.dirname(f), exist_ok=True)

    # write content to disk
    with open(f, "wb") as file:
        file.write(content)
