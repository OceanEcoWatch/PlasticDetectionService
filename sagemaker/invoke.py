import boto3

ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
CONTENT_TYPE = "application/octet-stream"

runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")

with open(
    "/Users/marc.leerink/dev/PlasticDetectionService/images/5cb12a6cbd6df0865947f21170bc432a/response.tiff",
    "rb",
) as f:
    payload = f.read()


response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME, ContentType=CONTENT_TYPE, Body=payload
)
print(response)
