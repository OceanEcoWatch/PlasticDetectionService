import base64
import json
import logging
from abc import ABC, abstractmethod

import runpod
from runpod import Endpoint

from src import config

LOGGER = logging.getLogger(__name__)


class BaseInferenceCallback(ABC):
    @abstractmethod
    def __call__(self, payload: bytes) -> bytes:
        """Perform inference on the given payload."""
        pass


class RunpodInferenceCallback(BaseInferenceCallback):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def __call__(self, payload: bytes) -> bytes:
        encoded_payload = base64.b64encode(payload).decode("utf-8")

        request_input = {"input": {"image": encoded_payload}}

        runpod.api_key = config.RUNPOD_API_KEY
        endpoint = Endpoint(self.endpoint_url)

        run_response = endpoint.run_sync(request_input, timeout=120)

        max_retries = 3
        retries = 0
        if not run_response:
            LOGGER.info("Retrying inference")
            retries += 1
            return self.__call__(payload)
        if retries > max_retries:
            raise RuntimeError("Max retries exceeded. Inference failed.")

        pred_bytes = base64.b64decode(json.loads(run_response)["prediction"])  # type: ignore

        return pred_bytes
