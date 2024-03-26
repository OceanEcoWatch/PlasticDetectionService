import io
import logging

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

LOGGER = logging.getLogger(__name__)


def stream_to_s3(
    data_stream: io.BytesIO,
    bucket_name: str,
    object_name: str,
) -> str:
    """Uploads a file to an S3 bucket and returns the URL to the uploaded file"""
    s3 = boto3.client("s3")
    try:
        s3.upload_fileobj(data_stream, bucket_name, object_name)
        LOGGER.info("File uploaded to s3://%s/%s", bucket_name, object_name)
        return f"s3://{bucket_name}/{object_name}"
    except NoCredentialsError as e:
        LOGGER.error("No AWS credentials found: %s", e)
        raise e
    except ClientError as e:
        LOGGER.error("Unexpected error: %s", e)
        raise e
