import requests


class EmbeddingClient:

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:

        url = f"{self.base_url}/api/embeddings"

        payload = {
            "model": self.model_name,
            "prompt": text
        }

        response = requests.post(url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")

        return response.json()["embedding"]