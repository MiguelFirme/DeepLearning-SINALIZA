"""Executa uma previsão de sinal em um único vídeo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import DIR_MODELOS  # noqa: E402
from sinaliza.landmarks import extrair_video  # noqa: E402
from sinaliza.sequencias import redimensionar_sequencia  # noqa: E402


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("video", type=Path)
    analisador.add_argument("--diretorio-modelo", type=Path, default=DIR_MODELOS)
    analisador.add_argument(
        "--modelo",
        type=Path,
        help="Arquivo .keras. Por padrão usa o último modelo treinado.",
    )
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    diretorio = argumentos.diretorio_modelo.resolve()
    if argumentos.modelo:
        caminho_modelo = argumentos.modelo.resolve()
    else:
        metadados = diretorio / "modelo_atual.json"
        if metadados.is_file():
            nome_modelo = json.loads(metadados.read_text(encoding="utf-8"))["arquivo"]
            caminho_modelo = diretorio / nome_modelo
        else:
            caminho_modelo = diretorio / "modelo_gru.keras"
    modelo = keras.models.load_model(caminho_modelo)
    rotulos = json.loads((diretorio / "rotulos.json").read_text(encoding="utf-8"))
    comprimento = int(modelo.input_shape[1])
    sequencia = redimensionar_sequencia(extrair_video(argumentos.video.resolve()), comprimento)
    probabilidades = modelo.predict(sequencia[np.newaxis, ...], verbose=0)[0]
    melhor_indice = int(np.argmax(probabilidades))
    print(f"Sinal: {rotulos[melhor_indice]}")
    print(f"Confiança: {probabilidades[melhor_indice]:.2%}")


if __name__ == "__main__":
    main()
