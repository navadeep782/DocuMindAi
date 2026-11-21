
from functools import lru_cache

from sentence_transformers import SentenceTransformer


# Pre-trained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the embedding model only once.
    Reuse it for subsequent embedding requests.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text strings into numerical embeddings.

    Input:
        ["TCP provides reliable communication.", "UDP is connectionless."]

    Output:
        [
            [0.12, -0.34, ...],  # 384 dimensions
            [0.45, 0.21, ...]    # 384 dimensions
        ]
    """

    # Handle empty input
    if not texts:
        return []

    # Validate input
    if any(not text.strip() for text in texts):
        raise ValueError(
            "Texts must not contain empty or whitespace-only strings."
        )

    # Load the model
    model = _get_model()

    # Generate embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Convert NumPy arrays into Python lists
    return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
