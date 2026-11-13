
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import ask_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    """
    Answer a question using all indexed documents.
    """

    try:
        result = ask_question(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("RAG service configuration or generation error")
        raise HTTPException(
            status_code=503,
            detail="The answer generation service is unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error while answering question")
        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer at this time.",
        ) from exc
