"""Baixa o V-LIBRASIL do Kaggle para a pasta de dados brutos do projeto."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import kagglehub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import DIR_VLIBRASIL_BRUTO  # noqa: E402
from sinaliza.datasets.v_librasil import descobrir_amostras  # noqa: E402


DATASET_KAGGLE = "davimedio01/v-librasil"


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--destino", type=Path, default=DIR_VLIBRASIL_BRUTO)
    analisador.add_argument(
        "--forcar",
        action="store_true",
        help="Baixa novamente mesmo quando o KaggleHub já possui uma cópia local.",
    )
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    destino = argumentos.destino.resolve()
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {DATASET_KAGGLE} para {destino}...")
    caminho = Path(
        kagglehub.dataset_download(
            DATASET_KAGGLE,
            output_dir=str(destino),
            force_download=argumentos.forcar,
        )
    ).resolve()
    amostras = descobrir_amostras(caminho)
    print(f"Download concluído: {len(amostras)} vídeos identificados em {caminho}")


if __name__ == "__main__":
    main()
