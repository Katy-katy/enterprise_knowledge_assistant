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
@app.post("/ask_stream")
async def ask_stream(request: AskRequest):

    async def stream():

        # 1️⃣ Retrieval
        query_embedding = embedding_client.embed(request.question)
        results = orchestrator.vector_store.search(query_embedding, top_k=15)

        candidate_docs = [doc for _, doc in results]

        reranked_docs = orchestrator.rerank(
            request.question,
            [doc["text"] for doc in candidate_docs],
            top_k=3
        )

        final_docs = [
            doc for doc in candidate_docs
            if doc["text"] in reranked_docs
        ]

        # 2️⃣ Build context
        context = "\n\n".join([doc["text"] for doc in final_docs])

        prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{request.question}

Answer:
"""

        # 3️⃣ STREAM TOKENS
        for token in llm_client.generate_stream(prompt):
            yield json.dumps({
                "type": "token",
                "content": token
            }) + "\n"

        # 4️⃣ SEND CITATIONS AT END
        citations = [
            f"Page {doc['page']} (chunk {doc['chunk_id']})"
            for doc in final_docs
        ]
        print("Citations========================================:", citations)

        yield json.dumps({
            "type": "citations",
            "content": citations
        }) + "\n"

        # 5️⃣ DONE SIGNAL
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/json")



@app.get("/health")
def health():
    return {"status": "ok"}
