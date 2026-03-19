import time
import asyncio

from app.llm.base import LLMClient
from app.models.schemas import AskResponse
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import VectorStore


class Orchestrator:
    def __init__(self, llm_client: LLMClient, embedding_client: EmbeddingClient):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = VectorStore()

    async def handle_question(self, question: str) -> AskResponse:
        start_time = time.time()

        query_embedding = self.embedding_client.embed(question)
        results = self.vector_store.search(query_embedding, top_k=3)
        retrieved_chunks = [doc for _, doc in results]

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        loop = asyncio.get_event_loop()

        answer = await loop.run_in_executor(
            None,
            self.llm_client.generate,
            prompt
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return AskResponse(
            answer=answer,
            citations=retrieved_chunks,
            confidence=0.0,
            latency_ms=latency_ms,
            model=self.llm_client.model_name,
        )