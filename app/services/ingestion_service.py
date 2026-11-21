
from uuid import uuid4

from app.services.pdf_service import extract_pages, chunk_pages
from app.services.embedding_service import embed_texts
from app.services.vector_store import get_vector_store


def ingest_pdf(
    pdf_bytes: bytes,
    source_filename: str,
) -> dict:
    """
    Process a PDF and store its chunks and embeddings in ChromaDB.

    Pipeline:
    PDF -> Text Extraction -> Chunking -> Embeddings -> ChromaDB
    """

    if not pdf_bytes:
        raise ValueError("PDF file cannot be empty.")

    if not source_filename.strip():
        raise ValueError("Source filename cannot be empty.")

    # Step 1: Extract text page by page.
    pages = extract_pages(pdf_bytes)

    # Step 2: Split page text into chunks.
    chunks = chunk_pages(
        pages=pages,
        source_filename=source_filename,
    )

    if not chunks:
        raise ValueError("No text chunks were generated.")

    # Step 3: Generate embeddings.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding count does not match chunk count."
        )

    # Step 4: Generate a unique document ID.
    document_id = f"doc_{uuid4().hex}"

    # Step 5: Store chunks and embeddings.
    vector_store = get_vector_store()

    stored_chunks = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
    )

    # Step 6: Return ingestion summary.
    return {
        "document_id": document_id,
        "filename": source_filename,
        "pages_with_text": len(pages),
        "chunk_count": len(chunks),
        "stored_chunks": stored_chunks,
        "status": "indexed",
    }
