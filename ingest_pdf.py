from app.retrieval.pdf_loader import load_pdf_with_pages
from app.retrieval.chunker import chunk_document
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import VectorStore

embedder = EmbeddingClient(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)

store = VectorStore()

pages = load_pdf_with_pages("sample.pdf")

for page in pages:
    chunks = chunk_document(page["text"])
    print("Chunks:", len(chunks))
    for i, chunk in enumerate(chunks):

        embedding = embedder.embed(chunk)

        metadata = {
            "text": chunk,
            "page": page["page"],
            "chunk_id": f"page_{page['page']}_chunk_{i}"
        }

        store.add(embedding, metadata)

store.save()

print("Ingestion complete.")