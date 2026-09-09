"""Configurações e caminhos utilizados pelo projeto."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ_PROJETO / ".env")

DIR_DATASETS = RAIZ_PROJETO / "datasets"
DIR_VLIBRASIL_PROCESSADO = DIR_DATASETS / "processados" / "v_librasil"
DIR_ALFABETO_PROCESSADO = DIR_DATASETS / "processados" / "alfabeto_libras"
DIR_NORMALIZADOS = DIR_DATASETS / "normalizados"
DIR_MODELOS = RAIZ_PROJETO / "models"


def diretorio_configurado(variavel: str) -> Path:
    """Retorna um diretório de dataset configurado no arquivo ``.env``."""
    caminho_configurado = os.getenv(variavel, "").strip()
    if not caminho_configurado:
        raise ValueError(
            f"{variavel} não foi definido. Copie .env.example para .env e "
            "informe o caminho, ou use --entrada."
        )
    return Path(caminho_configurado).expanduser().resolve()


def diretorio_vlibrasil() -> Path:
    """Retorna o diretório do V-LIBRASIL, dataset principal."""
    return diretorio_configurado("VLIBRASIL_DIR")


def diretorio_alfabeto() -> Path:
    """Retorna o diretório opcional do dataset de alfabeto."""
    return diretorio_configurado("ALFABETO_LIBRAS_DIR")
