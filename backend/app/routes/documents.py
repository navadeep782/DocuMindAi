
from fastapi import APIRouter, File, HTTPException, UploadFile
import pymupdf

from app.services.ingestion_service import ingest_pdf as process_pdf

router = APIRouter(prefix="/api/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    # 1. Validate file extension.
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # 2. Read the file with an upload size limit.
    pdf_bytes = await file.read(MAX_FILE_SIZE + 1)

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB upload limit.",
        )

    # 3. Check PDF file signature.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    # 4. Run the complete ingestion pipeline.
    try:
        result = process_pdf(
            pdf_bytes=pdf_bytes,
            source_filename=filename,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except pymupdf.FileDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded PDF.",
        ) from exc

    except Exception as exc:
        # Log the actual exception on the server for debugging.
        print(f"PDF ingestion failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the PDF.",
        ) from exc
