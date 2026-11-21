
from app.services.embedding_service import embed_texts
from app.services.vector_store import get_vector_store


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant chunks across ALL uploaded documents.

    Pipeline:
    Query -> Embedding -> Global ChromaDB Search -> Relevant Chunks
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Generate query embedding.
    query_embedding = embed_texts([query])[0]

    if not query_embedding:
        raise ValueError("Query embedding could not be generated.")

    # Access vector store.
    vector_store = get_vector_store()

    # Search across all indexed documents.
    results = vector_store.search_chunks(
        query_embedding=query_embedding,
        n_results=top_k,
    )

    # Format retrieved chunks.
    retrieved_chunks = []

    for result in results:
        distance = result["distance"]
        metadata = result["metadata"]

        retrieved_chunks.append({
            "text": result["text"],
            "source": metadata["source"],
            "page_number": metadata["page_number"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance,
            "similarity_score": round(1 - distance, 4),
        })

    return retrieved_chunks
