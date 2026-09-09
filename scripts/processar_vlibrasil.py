"""Cataloga e processa o V-LIBRASIL, dataset principal do projeto."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import (  # noqa: E402
    DIR_NORMALIZADOS,
    DIR_VLIBRASIL_PROCESSADO,
    diretorio_vlibrasil,
)
from sinaliza.datasets.v_librasil import descobrir_amostras, gravar_manifesto  # noqa: E402
from sinaliza.processamento import processar_vlibrasil  # noqa: E402


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--entrada", type=Path, help="Diretório bruto do V-LIBRASIL")
    analisador.add_argument("--saida", type=Path, default=DIR_VLIBRASIL_PROCESSADO)
    analisador.add_argument("--comprimento-sequencia", type=int, default=60)
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    entrada = argumentos.entrada.resolve() if argumentos.entrada else diretorio_vlibrasil()
    amostras = descobrir_amostras(entrada)
    gravar_manifesto(amostras, DIR_NORMALIZADOS / "v_librasil.csv")
    print(f"Foram catalogados {len(amostras)} vídeos do V-LIBRASIL.")
    processar_vlibrasil(
        amostras,
        argumentos.saida.resolve(),
        argumentos.comprimento_sequencia,
    )


if __name__ == "__main__":
    main()

