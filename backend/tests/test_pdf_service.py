
import pymupdf
import pytest

from app.services.pdf_service import extract_pages, chunk_pages


def make_test_pdf() -> bytes:
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "KnowledgeHub AI processes documents for research.",
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_extract_pages_preserves_page_number():
    pages = extract_pages(make_test_pdf())

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "KnowledgeHub AI" in pages[0]["text"]


def test_chunking_preserves_source_metadata():
    pages = extract_pages(make_test_pdf())

    chunks = chunk_pages(
        pages,
        source_filename="sample.pdf",
        chunk_size=30,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == 0
    assert all(c["source"] == "sample.pdf" for c in chunks)
    assert all(c["page_number"] == 1 for c in chunks)
    assert all(c["text"].strip() for c in chunks)


def test_empty_pdf_has_no_extractable_text():
    document = pymupdf.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pages(pdf_bytes)
