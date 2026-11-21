
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.gemini_service import generate_grounded_answer


def ask_question(
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Execute the global RAG pipeline.

    Question -> Embedding -> Global Retrieval -> Context
    -> Gemini -> Answer
    """

    # Step 1: Validate input.
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Step 2: Retrieve relevant chunks across ALL documents.
    retrieved_chunks = retrieve_relevant_chunks(
        query=question,
        top_k=top_k,
    )

    # Step 3: Generate an answer using retrieved context.
    answer = generate_grounded_answer(
        question=question,
        context_chunks=retrieved_chunks,
    )

    # Step 4: Prepare source references.
    sources = [
        {
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "similarity_score": chunk["similarity_score"],
        }
        for chunk in retrieved_chunks
    ]

    # Step 5: Return structured response.
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(retrieved_chunks),
    }

