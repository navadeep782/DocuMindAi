
from io import BytesIO

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract non-empty text from a PDF, preserving page numbers."""

    pages = []

    with pymupdf.open(
        stream=BytesIO(pdf_bytes),
        filetype="pdf",
    ) as document:
        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True).strip()

            if text:
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or empty."
        )

    return pages


def chunk_pages(
    pages: list[dict],
    source_filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Split page text while preserving source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[
                {
                    "source": source_filename,
                    "page_number": page["page_number"],
                }
            ],
        )

        for document in page_chunks:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": document.page_content,
                    "source": document.metadata["source"],
                    "page_number": document.metadata["page_number"],
                    "start_index": document.metadata["start_index"],
                }
            )

    return chunks
