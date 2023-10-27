import io

import boto3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
CONTENT_TYPE = "application/octet-stream"


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


if __name__ == "__main__":
    with open(
        "/Users/marc.leerink/dev/PlasticDetectionService/images/5cb12a6cbd6df0865947f21170bc432a/response.tiff",
        "rb",
    ) as f:
        payload = f.read()
    predictions = invoke(ENDPOINT_NAME, CONTENT_TYPE, payload)

    img = np.array(Image.open(io.BytesIO(predictions)))
    plt.imshow(img)
    plt.show()
