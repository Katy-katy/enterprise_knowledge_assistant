from app.retrieval.pdf_loader import load_pdf
from app.retrieval.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import VectorStore

embedder = EmbeddingClient(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)

store = VectorStore()

text = load_pdf("sample.pdf")

chunks = chunk_text(text)

print("Chunks:", len(chunks))

for chunk in chunks:
    embedding = embedder.embed(chunk)
    store.add(embedding, chunk)

store.save()

print("Ingestion complete.")