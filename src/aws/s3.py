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


def download_from_s3(
    bucket_name: str,
    object_name: str,
) -> bytes:
    """Downloads a file from an S3 bucket and returns the file content"""
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        LOGGER.info("File downloaded from s3://%s/%s", bucket_name, object_name)
        return response["Body"].read()
    except NoCredentialsError as e:
        LOGGER.error("No AWS credentials found: %s", e)
        raise e
    except ClientError as e:
        LOGGER.error("Unexpected error: %s", e)
        raise e


def get_folder_contents(
    bucket_name: str,
    folder_name: str,
):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_name):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            yield key
