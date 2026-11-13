
import pytest

from app.services import ingestion_service


class FakeVectorStore:
    def __init__(self):
        self.saved_chunks = []
        self.saved_embeddings = []
        self.saved_document_id = None

    def add_chunks(self, chunks, embeddings, document_id):
        self.saved_chunks = chunks
        self.saved_embeddings = embeddings
        self.saved_document_id = document_id
        return len(chunks)


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake_store = FakeVectorStore()

    sample_pages = [
        {
            "page_number": 1,
            "text": "TCP is a reliable protocol."
        },
        {
            "page_number": 2,
            "text": "UDP is a connectionless protocol."
        },
    ]

    sample_chunks = [
        {
            "chunk_id": 0,
            "text": "TCP is a reliable protocol.",
            "source": "network.pdf",
            "page_number": 1,
            "start_index": 0,
        },
        {
            "chunk_id": 1,
            "text": "UDP is a connectionless protocol.",
            "source": "network.pdf",
            "page_number": 2,
            "start_index": 0,
        },
    ]

    monkeypatch.setattr(
        ingestion_service,
        "extract_pages",
        lambda pdf_bytes: sample_pages,
    )

    monkeypatch.setattr(
        ingestion_service,
        "chunk_pages",
        lambda pages, source_filename: sample_chunks,
    )

    monkeypatch.setattr(
        ingestion_service,
        "embed_texts",
        lambda texts: [[0.1, 0.2], [0.3, 0.4]],
    )

    monkeypatch.setattr(
        ingestion_service,
        "get_vector_store",
        lambda: fake_store,
    )

    return fake_store


def test_ingest_pdf_success(fake_pipeline):
    result = ingestion_service.ingest_pdf(
        pdf_bytes=b"fake-pdf-content",
        source_filename="network.pdf",
    )

    assert result["filename"] == "network.pdf"
    assert result["pages_with_text"] == 2
    assert result["chunk_count"] == 2
    assert result["stored_chunks"] == 2
    assert result["status"] == "indexed"

    assert result["document_id"].startswith("doc_")
    assert fake_pipeline.saved_document_id == result["document_id"]
    assert len(fake_pipeline.saved_chunks) == 2
    assert len(fake_pipeline.saved_embeddings) == 2


def test_ingest_pdf_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        ingestion_service.ingest_pdf(
            pdf_bytes=b"",
            source_filename="network.pdf",
        )


def test_ingest_pdf_rejects_empty_filename():
    with pytest.raises(ValueError, match="filename"):
        ingestion_service.ingest_pdf(
            pdf_bytes=b"fake-pdf-content",
            source_filename="   ",
        )


def test_ingest_pdf_rejects_embedding_count_mismatch(
    fake_pipeline,
    monkeypatch,
):
    monkeypatch.setattr(
        ingestion_service,
        "embed_texts",
        lambda texts: [[0.1, 0.2]],
    )

    with pytest.raises(ValueError, match="Embedding count"):
        ingestion_service.ingest_pdf(
            pdf_bytes=b"fake-pdf-content",
            source_filename="network.pdf",
        )
