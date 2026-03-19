from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.core.orchestrator import Orchestrator
from app.llm.ollama_client import OllamaClient
from app.models.schemas import AskRequest, AskResponse
from app.retrieval.embeddings import EmbeddingClient


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Internal LLM-powered knowledge retrieval service",
    version="0.1.0",
)

llm_client = OllamaClient(base_url="http://localhost:11434", model_name="phi")
embedding_client = EmbeddingClient(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)
orchestrator = Orchestrator(llm_client=llm_client, embedding_client=embedding_client)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        return await orchestrator.handle_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-stream")
def ask_stream(request: AskRequest):
    def stream():
        for chunk in llm_client.generate_stream(request.question):
            data = json.loads(chunk)
            if "response" in data:
                yield data["response"]

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}
