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
    async def stream():

        # reuse retrieval logic
        query_embedding = embedding_client.embed(request.question)
        results = orchestrator.vector_store.search(query_embedding, top_k=3)

        context = "\n\n".join([doc for _, doc in results])

        prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{request.question}

Answer:
"""

        for chunk in llm_client.generate_stream(prompt):
            data = json.loads(chunk)
            if "response" in data:
                yield data["response"]

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}
