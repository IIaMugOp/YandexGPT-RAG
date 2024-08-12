import requests
import json
from typing import Optional, List, Mapping, Any
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class YandexLLM(LLM):
    api_key: str = None
    folder_id: str = None
    max_tokens : int = 1500
    temperature : float = 0.3

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {'Authorization': f'Api-Key {self.api_key}', "x-folder-id": f"{self.folder_id}"}
        req_completion = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite/latest",
            "completionOptions": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": [
                {
                    "role": "system",
                    "text": f"{prompt}"
                }
            ]
        }

        answer = requests.post(url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion", #("https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
                            headers=headers, json=req_completion)

        res = json.loads(answer.text)
        return res["result"]["alternatives"][0]["message"]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": "YandexGPT-RAG"}

    @property
    def _llm_type(self) -> str:
        return "yagpt"