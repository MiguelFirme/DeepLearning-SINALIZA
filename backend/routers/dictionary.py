"""Endpoints do dicionário de sinais.

O conteúdo vem da pasta  backend/static/dictionary/<Categoria>/<arquivo>,  que espelha
a pasta "Dicionário - SINALIZA" do SharePoint (uma subpasta por categoria):

    Alfabeto/A.png                        -> imagem, nome "A"
    Educação/Escola_Articulador1.mp4      -> vídeo,  nome "Escola"

A pasta é preenchida por scripts/prepare_dictionary_videos.py. Se estiver vazia, o
dicionário cai no catálogo antigo (label_map.json ou lista padrão), sem vídeos.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Query

from backend.schemas.models import DictionaryEntry, DictionaryResponse

router = APIRouter(prefix="/dictionary", tags=["dictionary"])

DICTIONARY_DIR = Path(__file__).resolve().parent.parent / "static" / "dictionary"
MEDIA_URL_PREFIX = "/api/media/dictionary"

VIDEO_EXTS = {".mp4", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
POSTER_SUFFIX = ".poster.jpg"  # frame de capa de cada vídeo: <nome>.poster.jpg
_ARTICULADOR_RE = re.compile(r"_Articulador(\d+)$", re.IGNORECASE)


def _fold(text: str) -> str:
    """Minúsculas e sem acento: 'Ônibus' -> 'onibus' (busca e ordenação)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return ascii_text.lower()


def _natural_key(text: str) -> list:
    """Ordem natural: '2' antes de '10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", _fold(text))]


def _media_url(category: str, filename: str) -> str:
    return f"{MEDIA_URL_PREFIX}/{quote(category)}/{quote(filename)}"


def _load_from_folder() -> list[DictionaryEntry]:
    """Monta o dicionário a partir das subpastas de DICTIONARY_DIR."""
    if not DICTIONARY_DIR.is_dir():
        return []

    # (categoria, nome) -> (prioridade, entrada). Se a palavra tiver mais de um
    # articulador, fica o de menor número; vídeo ganha de imagem no empate.
    best: dict[tuple[str, str], tuple[tuple[int, int], DictionaryEntry]] = {}

    for cat_dir in (d for d in DICTIONARY_DIR.iterdir() if d.is_dir()):
        category = cat_dir.name
        for file in cat_dir.iterdir():
            if not file.is_file() or file.name.lower().endswith(POSTER_SUFFIX):
                continue
            ext = file.suffix.lower()
            if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
                continue

            match = _ARTICULADOR_RE.search(file.stem)
            name = (file.stem[: match.start()] if match else file.stem).strip()
            rank = (int(match.group(1)) if match else 0, 0 if ext in VIDEO_EXTS else 1)
            if not name:
                continue

            entry = DictionaryEntry(sign_id=0, name=name, category=category)
            if ext in VIDEO_EXTS:
                entry.example_video_url = _media_url(category, file.name)
                poster = cat_dir / f"{file.stem}{POSTER_SUFFIX}"
                if poster.is_file():
                    entry.poster_url = _media_url(category, poster.name)
            else:
                entry.image_url = _media_url(category, file.name)

            key = (category, _fold(name))
            if key not in best or rank < best[key][0]:
                best[key] = (rank, entry)

    entries = [e for _rank, e in best.values()]
    entries.sort(key=lambda e: (_natural_key(e.category), _natural_key(e.name)))
    for i, entry in enumerate(entries):
        entry.sign_id = i  # ids estáveis enquanto os arquivos não mudarem
    return entries


# ---------- Catálogo antigo (usado só se a pasta do dicionário estiver vazia) ----------

DEFAULT_CATEGORIES = {
    "Saudações": ["Oi", "Bom dia", "Boa tarde", "Boa noite", "Obrigado", "Por favor", "Desculpa", "Tchau"],
    "Família": ["Pai", "Mãe", "Irmão", "Irmã", "Filho", "Filha", "Avô", "Avó"],
    "Tempo": ["Hoje", "Amanhã", "Ontem", "Semana", "Mês", "Ano", "Hora", "Minuto"],
    "Emoções": ["Feliz", "Triste", "Bravo", "Medo", "Amor", "Saudade", "Surpresa"],
    "Educação": ["Escola", "Professor", "Aluno", "Livro", "Estudar", "Aprender", "Aula"],
}


def _load_fallback_catalog() -> list[DictionaryEntry]:
    entries = []
    sign_id = 0

    label_map_path = Path("data/processed/label_map.json")
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        for name in sorted(label_map.keys()):
            category = "Outros"
            for cat, signs in DEFAULT_CATEGORIES.items():
                if name in signs:
                    category = cat
                    break
            entries.append(DictionaryEntry(sign_id=label_map[name], name=name, category=category))
        return entries

    for category, signs in DEFAULT_CATEGORIES.items():
        for name in signs:
            entries.append(DictionaryEntry(sign_id=sign_id, name=name, category=category))
            sign_id += 1
    return entries


def _load_dictionary() -> list[DictionaryEntry]:
    return _load_from_folder() or _load_fallback_catalog()


def _categories_of(entries: list[DictionaryEntry]) -> list[str]:
    return list(dict.fromkeys(e.category for e in entries))  # mantém a ordem


@router.get("", response_model=DictionaryResponse)
async def get_dictionary(
    category: str | None = Query(None, description="Filtrar por categoria"),
    search: str | None = Query(None, description="Buscar por nome (ignora acentos)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Retorna o dicionário de sinais com filtros."""
    all_entries = _load_dictionary()
    entries = all_entries

    if category:
        entries = [e for e in entries if _fold(e.category) == _fold(category)]
    if search:
        q = _fold(search)
        entries = [e for e in entries if q in _fold(e.name)]

    total = len(entries)
    entries = entries[offset : offset + limit]
    return DictionaryResponse(entries=entries, total=total, categories=_categories_of(all_entries))


@router.get("/categories")
async def get_categories():
    """Lista categorias disponíveis."""
    return {"categories": _categories_of(_load_dictionary())}
