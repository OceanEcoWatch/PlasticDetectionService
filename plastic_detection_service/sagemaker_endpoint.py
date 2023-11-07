import boto3


def invoke(endpoint_name: str, content_type: str, payload: bytes) -> bytes:
    runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload,
        Accept=content_type,
    )
    predictions = response["Body"].read()
    return predictions
