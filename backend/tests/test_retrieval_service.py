
import pytest

from app.services import retrieval_service


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.received_query_embedding = None
        self.received_n_results = None
        self.received_document_id = None

    def search_chunks(
        self,
        query_embedding,
        n_results=5,
        document_id=None,
    ):
        self.received_query_embedding = query_embedding
        self.received_n_results = n_results
        self.received_document_id = document_id

        return self.results


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeVectorStore()

    monkeypatch.setattr(
        retrieval_service,
        "get_vector_store",
        lambda: store,
    )

    monkeypatch.setattr(
        retrieval_service,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    return store


def test_retrieve_relevant_chunks_success(fake_store):
    fake_store.results = [
        {
            "text": "ACID stands for Atomicity, Consistency, Isolation and Durability.",
            "metadata": {
                "source": "DBMS_Full_Notes.pdf",
                "page_number": 12,
                "chunk_id": 4,
            },
            "distance": 0.15,
        }
    ]

    results = retrieval_service.retrieve_relevant_chunks(
        query="Explain ACID properties",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["source"] == "DBMS_Full_Notes.pdf"
    assert results[0]["page_number"] == 12
    assert results[0]["similarity_score"] == 0.85


def test_retrieve_passes_top_k_and_document_id(fake_store):
    retrieval_service.retrieve_relevant_chunks(
        query="Explain normalization",
        top_k=3,
        document_id="doc_123",
    )

    assert fake_store.received_query_embedding == [0.1, 0.2, 0.3]
    assert fake_store.received_n_results == 3
    assert fake_store.received_document_id == "doc_123"


def test_empty_query_raises_error(fake_store):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieval_service.retrieve_relevant_chunks("")


def test_whitespace_query_raises_error(fake_store):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieval_service.retrieve_relevant_chunks("   ")


def test_invalid_top_k_raises_error(fake_store):
    with pytest.raises(ValueError, match="top_k must be positive"):
        retrieval_service.retrieve_relevant_chunks(
            query="Explain DBMS",
            top_k=0,
        )


def test_empty_document_id_raises_error(fake_store):
    with pytest.raises(ValueError, match="Document ID cannot be empty"):
        retrieval_service.retrieve_relevant_chunks(
            query="Explain DBMS",
            document_id="   ",
        )


def test_no_matching_chunks_returns_empty_list(fake_store):
    fake_store.results = []

    results = retrieval_service.retrieve_relevant_chunks(
        query="Explain an unrelated topic",
    )

    assert results == []
