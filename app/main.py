

from fastapi import FastAPI

from app.routes.documents import router as documents_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title="DocuMind AI",
    description="An intelligent document research assistant powered by RAG.",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to DocuMind AI API",
        "status": "running",
    }


app.include_router(documents_router)
app.include_router(chat_router)
