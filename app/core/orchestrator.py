import time

from app.llm.base import LLMClient
from app.models.schemas import AskResponse


class Orchestrator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def handle_question(self, question: str) -> AskResponse:
        start_time = time.time()

        answer = self.llm_client.generate(question)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return AskResponse(
            answer=answer,
            citations=[],
            confidence=0.5,
            latency_ms=latency_ms,
            model=self.llm_client.model_name,
        )