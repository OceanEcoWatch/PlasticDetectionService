import base64
import json
from abc import ABC, abstractmethod

import runpod
from runpod import Endpoint

from src import config


class BaseInferenceCallback(ABC):
    @abstractmethod
    def __call__(self, payload: bytes) -> bytes:
        """Perform inference on the given payload."""
        pass


class LocalInferenceCallback(BaseInferenceCallback):
    def __call__(self, payload: bytes) -> bytes:
        from .sagemaker_model.code.inference import (
            input_fn,
            model_fn,
            output_fn,
            predict_fn,
        )

        model = model_fn(".")
        array = input_fn(payload, "application/octet-stream")
        pred_array = predict_fn(array, model)
        return output_fn(pred_array, "application/octet-stream")


class RunpodInferenceCallback(BaseInferenceCallback):
    def __call__(self, payload: bytes) -> bytes:
        encoded_payload = base64.b64encode(payload).decode("utf-8")

        request_input = {"input": {"image": encoded_payload}}

        runpod.api_key = config.RUNPOD_API_KEY
        endpoint = Endpoint(config.RUNPOD_ENDPOINT_ID)

        run_response = endpoint.run_sync(request_input, timeout=60)
        pred_bytes = base64.b64decode(json.loads(run_response)["prediction"])
        return pred_bytes
