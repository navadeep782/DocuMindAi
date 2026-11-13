
import pytest

from app.services import rag_service


@pytest.fixture
def mock_rag_dependencies(monkeypatch):
    chunks = [
        {
            "text": "ACID stands for Atomicity, Consistency, Isolation and Durability.",
            "source": "DBMS_Full_Notes.pdf",
            "page_number": 12,
            "chunk_id": 4,
            "distance": 0.15,
            "similarity_score": 0.85,
        },
        {
            "text": "Atomicity ensures an all-or-nothing transaction.",
            "source": "DBMS_Full_Notes.pdf",
            "page_number": 13,
            "chunk_id": 5,
            "distance": 0.20,
            "similarity_score": 0.80,
        },
    ]

    calls = {}

    def fake_retrieve(query, top_k=5, document_id=None):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["document_id"] = document_id
        return chunks[:top_k]

    def fake_generate(question, context_chunks):
        calls["question"] = question
        calls["context_chunks"] = context_chunks
        return "ACID properties ensure transaction reliability."

    monkeypatch.setattr(
        rag_service,
        "retrieve_relevant_chunks",
        fake_retrieve,
    )

    monkeypatch.setattr(
        rag_service,
        "generate_grounded_answer",
        fake_generate,
    )

    return calls


def test_ask_question_success(mock_rag_dependencies):
    result = rag_service.ask_question(
        question="Explain ACID properties",
        top_k=2,
    )

    assert result["question"] == "Explain ACID properties"
    assert result["answer"] == (
        "ACID properties ensure transaction reliability."
    )
    assert result["retrieved_chunks"] == 2
    assert len(result["sources"]) == 2


def test_ask_question_passes_document_id(mock_rag_dependencies):
    rag_service.ask_question(
        question="Explain normalization",
        document_id="doc_123",
    )

    assert mock_rag_dependencies["document_id"] == "doc_123"


def test_ask_question_passes_top_k(mock_rag_dependencies):
    rag_service.ask_question(
        question="Explain transactions",
        top_k=1,
    )

    assert mock_rag_dependencies["top_k"] == 1


def test_empty_question_raises_error(mock_rag_dependencies):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag_service.ask_question("")


def test_invalid_top_k_raises_error(mock_rag_dependencies):
    with pytest.raises(ValueError, match="top_k must be positive"):
        rag_service.ask_question(
            question="Explain DBMS",
            top_k=0,
        )
