import requests
from typing import Iterator

from app.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self._model_name = model_name  # private attribute

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 150,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        print("+++++++++++++ payload: ", payload)
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
        data = response.json()
        print("+++++++++++++5: ", data)
        return data.get("response", "")

    def generate_stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 150,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    yield data