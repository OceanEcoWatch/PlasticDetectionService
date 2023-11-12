import boto3
from botocore.exceptions import ClientError


def invoke(endpoint_name: str, content_type: str, payload: bytes) -> bytes:
    runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload,
            Accept=content_type,
        )
        prediction = response["Body"].read()
        return prediction

    except ClientError as e:
        if "ThrottlingException" in str(e):
            print("ThrottlingException, retrying...")
            return invoke(endpoint_name, content_type, payload)

        else:
            print("Unexpected error: %s" % e)
            raise e
