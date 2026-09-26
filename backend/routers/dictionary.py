"""Endpoints do dicionário de sinais."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, Query

from backend.schemas.models import DictionaryEntry, DictionaryResponse

router = APIRouter(prefix="/dictionary", tags=["dictionary"])

# Categorias padrão de sinais
DEFAULT_CATEGORIES = {
    "Saudações": ["Oi", "Bom dia", "Boa tarde", "Boa noite", "Obrigado", "Por favor", "Desculpa", "Tchau"],
    "Família": ["Pai", "Mãe", "Irmão", "Irmã", "Filho", "Filha", "Avô", "Avó"],
    "Tempo": ["Hoje", "Amanhã", "Ontem", "Semana", "Mês", "Ano", "Hora", "Minuto"],
    "Emoções": ["Feliz", "Triste", "Bravo", "Medo", "Amor", "Saudade", "Surpresa"],
    "Educação": ["Escola", "Professor", "Aluno", "Livro", "Estudar", "Aprender", "Aula"],
}


def _load_dictionary() -> list[DictionaryEntry]:
    """Carrega dicionário de sinais do label_map ou arquivo estático."""
    entries = []
    sign_id = 0

    # Tentar carregar do label_map real
    label_map_path = Path("data/processed/label_map.json")
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        for name in sorted(label_map.keys()):
            # Encontrar categoria
            category = "Outros"
            for cat, signs in DEFAULT_CATEGORIES.items():
                if name in signs:
                    category = cat
                    break
            entries.append(DictionaryEntry(sign_id=label_map[name], name=name, category=category))
        return entries

    # Fallback: usar categorias padrão
    for category, signs in DEFAULT_CATEGORIES.items():
        for name in signs:
            entries.append(DictionaryEntry(sign_id=sign_id, name=name, category=category))
            sign_id += 1
    return entries


@router.get("", response_model=DictionaryResponse)
async def get_dictionary(
    category: str | None = Query(None, description="Filtrar por categoria"),
    search: str | None = Query(None, description="Buscar por nome"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Retorna o dicionário de sinais com filtros."""
    entries = _load_dictionary()

    if category:
        entries = [e for e in entries if e.category.lower() == category.lower()]
    if search:
        q = search.lower()
        entries = [e for e in entries if q in e.name.lower()]

    total = len(entries)
    entries = entries[offset : offset + limit]
    categories = sorted(set(e.category for e in _load_dictionary()))

    return DictionaryResponse(entries=entries, total=total, categories=categories)


@router.get("/categories")
async def get_categories():
    """Lista categorias disponíveis."""
    return {"categories": list(DEFAULT_CATEGORIES.keys())}
