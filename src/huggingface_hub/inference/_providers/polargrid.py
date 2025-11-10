# Minimal code, needs review
from typing import Any, Dict, Optional, Union
from ._common import TaskProviderHelper, BaseTextGenerationTask, BaseConversationalTask
from ._types import RequestParameters

PROVIDER_NAME = "polargrid"
PROVIDER_BASE_URL = "https://api.polargrid.com"  # need to place actual base url here

#Need confirmation these are the correct methods
class PolarGridConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=PROVIDER_NAME, base_url=PROVIDER_BASE_URL)

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict:
        headers = {**headers, "Authorization": f"Bearer {api_key}"}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _prepare_route(self,mapped_model: str, api_key: str) -> str:
        return "/v1/chat/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, mapped_model: str
    ) -> Optional[Dict]:
        messages = inputs
        if isinstance(inputs, str):
            messages = [{"role": "user", "content": inputs}]
        body: Dict[str, Any] = {"model": mapped_model, "messages": messages}
        # Mapping common parameters
        if "max_new_tokens" in parameters:
            body["max_tokens"] = parameters["max_new_tokens"]
        if "temperature" in parameters:
            body["temperature"] = parameters["temperature"]
        if "top_p" in parameters:
            body["top_p"] = parameters["top_p"]
        if "stop_sequences" in parameters:
            body["stop"] = parameters["stop_sequences"]

        for k, v in parameters.items():
            if k not in {"max_new_tokens", "temperature", "top_p", "stop_sequences"}:
                body.setdefault(k, v)

        return body

    def get_response(
        self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None
    ) -> Any:
        if isinstance(response, bytes):
            raise ValueError("Unexpected bytes response for chat-completions")

        choices = response.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"]["content"]

        return str(response)

class PolarGridTextGeneration(BaseTextGenerationTask):
    def __init__(self) -> None:
        super().__init__(provider=PROVIDER_NAME, base_url=PROVIDER_BASE_URL)

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict:
        headers = {**headers, "Authorization": f"Bearer {api_key}"}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/completions"

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return {
            "model": mapped_model,
            "prompt": str(inputs),
            # Only include max_tokens if provided
            **({"max_tokens": parameters["max_new_tokens"]} if "max_new_tokens" in parameters else {})
        }

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        if isinstance(response, bytes):
            raise ValueError("Unexpected bytes response for text-generation")
        choices = response.get("choices", [])
        return choices[0]["text"] if choices and "text" in choices[0] else str(response)


class PolarGridFeatureExtraction(TaskProviderHelper):
    def __init__(self) -> None:
        super().__init__(provider=PROVIDER_NAME, base_url=PROVIDER_BASE_URL, task="feature-extraction")

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict:
        headers = {**headers, "Authorization": f"Bearer {api_key}"}
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/embeddings"

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
    return {
        "model": mapped_model,
        "input": inputs if isinstance(inputs, list) else [inputs],
    }

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        if isinstance(response, bytes):
            raise ValueError("Unexpected bytes response for feature-extraction")
        return response
    
