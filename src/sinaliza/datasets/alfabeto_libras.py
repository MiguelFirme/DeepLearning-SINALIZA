"""Adaptador do dataset auxiliar de alfabeto estático em Libras."""

from __future__ import annotations

import csv
from pathlib import Path

from sinaliza.datasets.amostras import AmostraImagem


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def descobrir_amostras(root: Path) -> list[AmostraImagem]:
    """Cataloga imagens usando a pasta imediatamente superior como classe."""
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório do alfabeto não encontrado: {root}")

    samples = [
        AmostraImagem(rotulo=path.parent.name, caminho=path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if not samples:
        raise ValueError(f"Nenhuma imagem de alfabeto foi encontrada em {root}")
    return samples


def gravar_manifesto(samples: list[AmostraImagem], destination: Path) -> None:
    """Grava o catálogo do dataset auxiliar sem misturá-lo ao V-LIBRASIL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "rotulo", "origem"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "dataset": "alfabeto_libras",
                    "rotulo": sample.rotulo,
                    "origem": str(sample.caminho.resolve()),
                }
            )
