import boto3

import sagemaker
from sagemaker import image_uris

model_path = "s3://sagemaker-studio-768912473174-0ryazmj34j9/model.tar.gz"
model_name = "MarineDebrisDetectorModel"
sagemaker_role = "arn:aws:iam::768912473174:role/service-role/AmazonSageMaker-ExecutionRole-20231017T155251"

region_name = "eu-central-1"
image_uri = image_uris.retrieve(
    framework="pytorch",
    region="eu-central-1",
    version="2.0.1",
    py_version="py310",
    image_scope="inference",
    instance_type="ml.c5.4xlarge",
)

# Initialize sagemaker session
sess = sagemaker.Session()


# Initialize boto3 sagemaker client
region = sess.boto_region_name
sm_client = boto3.client("sagemaker", region_name=region)

# Define Model specifications
modelpackage_inference_specification = {
    "InferenceSpecification": {
        "Containers": [{"Image": image_uri, "ModelDataUrl": model_path}],
        "SupportedContentTypes": ["application/x-image"],
        "SupportedResponseMIMETypes": ["application/x-image"],
    }
}


create_model_package_input_dict = {
    "ModelApprovalStatus": "PendingManualApproval",
}
create_model_package_input_dict.update(modelpackage_inference_specification)

# # Create Model in Sagemaker Studio Model Registry
# create_model_package_response = sm_client.create_model_package(
#     **create_model_package_input_dict,
# )


create_model_response = sm_client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": model_path,
            "Environment": {
                "SAGEMAKER_SUBMIT_DIRECTORY": model_path,
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_REGION": region_name,
                "LOG_LOCATION": "/tmp",
                "METRICS_LOCATION": "/tmp",
            },
        }
    ],
    ExecutionRoleArn=sagemaker_role,
)

print(create_model_response)
