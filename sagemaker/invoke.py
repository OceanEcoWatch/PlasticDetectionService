import io

import boto3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
CONTENT_TYPE = "application/octet-stream"

runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")

with open(
    "/Users/marc.leerink/dev/PlasticDetectionService/images/5cb12a6cbd6df0865947f21170bc432a/response.tiff",
    "rb",
) as f:
    payload = f.read()


response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType=CONTENT_TYPE,
    Body=payload,
    Accept=CONTENT_TYPE,
)
predictions = response["Body"].read()


# convert the byte array to a numpy array
img = np.array(Image.open(io.BytesIO(predictions)))
plt.imshow(img)
plt.show()
