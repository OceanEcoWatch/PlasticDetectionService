import boto3

ENDPOINT_NAME = "test"
CONTENT_TYPE = "application/x-image"

runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")

with open("response.tiff", "rb") as f:
    payload = f.read()


response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME, ContentType=CONTENT_TYPE, Body=payload
)
print(response["Body"].read())
