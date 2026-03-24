import time
import asyncio

from app.llm.base import LLMClient
from app.models.schemas import AskResponse
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import VectorStore


TOP_K_RETRIEVAL = 10
TOP_K_RERANKING = 3


class Orchestrator:
    def __init__(self, llm_client: LLMClient, embedding_client: EmbeddingClient):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = VectorStore()

    def rerank(self, question: str, documents: list[str], top_k: int = 3):

        scored = []

        for doc in documents:

            prompt = f"""
        Rate how relevant this document is to the question.

        Question:
        {question}

        Document:
        {doc}

        Score from 0 to 10:
        """

            score_text = self.llm_client.generate(prompt)

            try:
                score = float(score_text.strip().split()[0])
            except:
                score = 0.0

            scored.append((score, doc))

        scored.sort(reverse=True)

        return [doc for _, doc in scored[:top_k]]


    async def handle_question(self, question: str) -> AskResponse:
        start_time = time.time()

        query_embedding = self.embedding_client.embed(question)
        results = self.vector_store.search(query_embedding, top_k=TOP_K_RETRIEVAL)
        retrieved_chunks = [doc for _, doc in results]
        reranked_docs = self.rerank(question, candidate_docs, top_k=TOP_K_RERANKING)
        context = "\n\n".join(reranked_docs)
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