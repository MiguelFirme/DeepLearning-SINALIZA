"""Cataloga o dataset auxiliar de imagens do alfabeto em Libras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import DIR_NORMALIZADOS, diretorio_alfabeto  # noqa: E402
from sinaliza.datasets.alfabeto_libras import (  # noqa: E402
    descobrir_amostras,
    gravar_manifesto,
)


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--entrada", type=Path, help="Diretório bruto do alfabeto")
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    entrada = argumentos.entrada.resolve() if argumentos.entrada else diretorio_alfabeto()
    amostras = descobrir_amostras(entrada)
    destino = DIR_NORMALIZADOS / "alfabeto_libras.csv"
    gravar_manifesto(amostras, destino)
    print(f"Foram catalogadas {len(amostras)} imagens auxiliares em {destino}.")


if __name__ == "__main__":
    main()

