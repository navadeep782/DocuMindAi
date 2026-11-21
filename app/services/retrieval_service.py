
from app.services.embedding_service import embed_texts
from app.services.vector_store import get_vector_store


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a user query.

    Pipeline:
    Query -> Embedding -> ChromaDB Search -> Relevant Chunks
    """

    # Step 1: Validate the query.
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if document_id is not None and not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    # Step 2: Convert the query into an embedding.
    query_embedding = embed_texts([query])[0]

    if not query_embedding:
        raise ValueError("Query embedding could not be generated.")

    # Step 3: Access the vector store.
    vector_store = get_vector_store()

    # Step 4: Search for the most relevant chunks.
    results = vector_store.search_chunks(
        query_embedding=query_embedding,
        n_results=top_k,
        document_id=document_id,
    )

    # Step 5: Format the retrieved results.
    retrieved_chunks = []

    for result in results:
        distance = result["distance"]

        retrieved_chunks.append({
            "text": result["text"],
            "source": result["metadata"]["source"],
            "page_number": result["metadata"]["page_number"],
            "chunk_id": result["metadata"]["chunk_id"],
            "distance": distance,
            "similarity_score": round(1 - distance, 4),
        })

    return retrieved_chunks
