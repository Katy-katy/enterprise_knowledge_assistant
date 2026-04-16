import numpy as np
import json
from typing import List


class VectorStore:

    def __init__(self, path: str = "vector_store.json"):
        self.path = path
        self.embeddings: List[np.ndarray] = []
        self.documents: List[dict] = []

        self._load()

    def add(self, embedding: List[float], document: dict):
        self.embeddings.append(np.array(embedding))
        self.documents.append(document)

    def search(self, query_embedding: List[float], top_k: int = 3):

        query = np.array(query_embedding)

        scores = []

        for i, emb in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query, emb)
            scores.append((similarity, self.documents[i]))

        scores.sort(reverse=True)

        return scores[:top_k]

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save(self):

        data = {
            "documents": self.documents,
            "embeddings": [emb.tolist() for emb in self.embeddings]
        }

        with open(self.path, "w") as f:
            json.dump(data, f)

    def _load(self):

        try:
            with open(self.path, "r") as f:
                data = json.load(f)

                self.documents = data["documents"]
                self.embeddings = [np.array(e) for e in data["embeddings"]]

        except FileNotFoundError:
            pass