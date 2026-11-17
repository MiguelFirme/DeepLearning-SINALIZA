"""Endpoint que transforma uma lista de sinais em uma frase natural."""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.sentence import build_sentence

router = APIRouter(tags=["sentence"])


class SentenceRequest(BaseModel):
    words: list[str] = Field(..., min_length=1, max_length=50, description="Sinais reconhecidos, em ordem")


class SentenceResponse(BaseModel):
    sentence: str
    source: str  # "llm" ou "fallback"


@router.post("/sentence", response_model=SentenceResponse)
async def sentence(request: SentenceRequest):
    text, source = await build_sentence(request.words)
    return SentenceResponse(sentence=text, source=source)