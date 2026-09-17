"""Executa download, processamento e treinamento do V-LIBRASIL em sequência."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
SCRIPTS = RAIZ_PROJETO / "scripts"


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--sem-download",
        action="store_true",
        help="Usa os arquivos que já estão em datasets/brutos/v_librasil.",
    )
    analisador.add_argument("--forcar-download", action="store_true")
    analisador.add_argument("--comprimento-sequencia", type=int, default=60)
    analisador.add_argument("--epocas", type=int, default=30)
    analisador.add_argument("--tamanho-lote", type=int, default=8)
    analisador.add_argument(
        "--processos",
        type=int,
        help="Processos paralelos na extração (padrão automático, limitado a 4)",
    )
    analisador.add_argument(
        "--arquitetura", choices=("bilstm", "gru"), default="bilstm"
    )
    return analisador.parse_args()


def executar(*argumentos: str) -> None:
    comando = [sys.executable, *argumentos]
    print("\n>", " ".join(comando))
    subprocess.run(comando, cwd=RAIZ_PROJETO, check=True)


def main() -> None:
    argumentos = ler_argumentos()
    if not argumentos.sem_download:
        comando_download = [str(SCRIPTS / "baixar_vlibrasil.py")]
        if argumentos.forcar_download:
            comando_download.append("--forcar")
        executar(*comando_download)

    comando_processamento = [
        str(SCRIPTS / "processar_vlibrasil.py"),
        "--comprimento-sequencia",
        str(argumentos.comprimento_sequencia),
    ]
    if argumentos.processos is not None:
        comando_processamento.extend(["--processos", str(argumentos.processos)])
    executar(*comando_processamento)
    executar(
        str(SCRIPTS / "treinar.py"),
        "--epocas",
        str(argumentos.epocas),
        "--tamanho-lote",
        str(argumentos.tamanho_lote),
        "--arquitetura",
        argumentos.arquitetura,
    )


if __name__ == "__main__":
    main()
