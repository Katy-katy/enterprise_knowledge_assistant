import requests
from typing import Iterator
import json

from app.llm.base import LLMClient
from app.utils import stats


OLLAMA_NUM_THREAD=8
OLLAMA_TIMEOUT = 600


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
                "top_p": 0.9,
                "num_thread": OLLAMA_NUM_THREAD,
                "num_batch": 512,
                "num_ctx": 1024, # since we have a small prompt we do not need default ~4096 context length
                "keep_alive": "10m" # keep the connection alive for 10 minutes to avoid loading 

            }
        }
        try:
            response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
        data = response.json()
        stats(data)
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
                "top_p": 0.9,
                "num_thread": OLLAMA_NUM_THREAD,
                "num_batch": 512,
                "num_ctx": 1024, # since we have a small prompt we do not need default ~4096 context length
                "keep_alive": "10m" # keep the connection alive for 10 minutes to avoid loading 
            }
        }

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    if token:
                        yield token
                except:
                    continue