from app.retrieval.embeddings import EmbeddingClient

client = EmbeddingClient(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)

vector = client.embed("Kubernetes manages containerized applications")

print(len(vector))


'''

import unittest
from unittest.mock import patch, MagicMock

from app.retrieval.embeddings import EmbeddingClient


class TestEmbeddingClient(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "nomic-embed-text"
        self.client = EmbeddingClient(base_url=self.base_url, model_name=self.model_name)

    def test_init_stores_base_url_and_model_name(self):
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client.model_name, self.model_name)

    @patch("app.retrieval.embeddings.requests.post")
    def test_embed_returns_embedding_list(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, -0.2, 0.3],
        }
        mock_post.return_value = mock_response

        result = self.client.embed("hello world")

        self.assertEqual(result, [0.1, -0.2, 0.3])
        mock_post.assert_called_once_with(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": "hello world"},
        )

    @patch("app.retrieval.embeddings.requests.post")
    def test_embed_raises_on_non_200(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with self.assertRaises(Exception) as ctx:
            self.client.embed("hello")

        self.assertIn("Embedding failed", str(ctx.exception))
        self.assertIn("Internal Server Error", str(ctx.exception))
'''