"""Tipos comuns retornados pelos adaptadores de datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AmostraVideo:
    rotulo: str
    articulador_id: str
    caminho: Path


@dataclass(frozen=True)
class AmostraImagem:
    rotulo: str
    caminho: Path
