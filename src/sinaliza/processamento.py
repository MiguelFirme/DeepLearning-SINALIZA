"""Extração das sequências do V-LIBRASIL."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from sinaliza.datasets.amostras import AmostraVideo
from sinaliza.landmarks import extrair_video
from sinaliza.sequencias import redimensionar_sequencia


def _nome_caracteristica(amostra: AmostraVideo) -> str:
    identidade = str(amostra.caminho.resolve()).encode("utf-8")
    return f"{hashlib.sha1(identidade).hexdigest()[:16]}.npy"


def processar_vlibrasil(
    amostras: list[AmostraVideo], diretorio_saida: Path, comprimento_sequencia: int
) -> None:
    """Extrai landmarks dos vídeos e cria um manifesto para o treinamento."""
    diretorio_landmarks = diretorio_saida / "landmarks"
    diretorio_landmarks.mkdir(parents=True, exist_ok=True)
    linhas: list[dict[str, str | int]] = []
    total_falhas = 0

    for indice, amostra in enumerate(amostras, start=1):
        print(f"[{indice}/{len(amostras)}] {amostra.rotulo}: {amostra.caminho.name}")
        try:
            sequencia = redimensionar_sequencia(
                extrair_video(amostra.caminho), comprimento_sequencia
            )
            destino = diretorio_landmarks / _nome_caracteristica(amostra)
            np.save(destino, sequencia)
            linhas.append(
                {
                    "dataset": "v_librasil",
                    "rotulo": amostra.rotulo,
                    "articulador_id": amostra.articulador_id,
                    "origem": str(amostra.caminho.resolve()),
                    "caminho_landmarks": str(destino.resolve()),
                    "quadros": sequencia.shape[0],
                    "caracteristicas": sequencia.shape[1],
                }
            )
        except (OSError, RuntimeError, ValueError) as error:
            total_falhas += 1
            print(f"  AVISO: {error}")

    manifesto = diretorio_saida / "manifesto.csv"
    with manifesto.open("w", newline="", encoding="utf-8") as file:
        campos = [
            "dataset",
            "rotulo",
            "articulador_id",
            "origem",
            "caminho_landmarks",
            "quadros",
            "caracteristicas",
        ]
        escritor = csv.DictWriter(file, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"Processamento concluído: {len(linhas)} sucesso(s), {total_falhas} falha(s).")
