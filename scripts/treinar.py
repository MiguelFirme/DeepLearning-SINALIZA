"""Treina o baseline GRU com os landmarks processados do V-LIBRASIL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import DIR_MODELOS, DIR_VLIBRASIL_PROCESSADO  # noqa: E402
from sinaliza.modelo import construir_modelo_gru  # noqa: E402


def carregar_amostras(
    diretorio_processado: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifesto = diretorio_processado / "manifesto.csv"
    if not manifesto.is_file():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {manifesto}. Execute processar_vlibrasil.py primeiro."
        )

    with manifesto.open("r", encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise ValueError("O manifesto processado não contém amostras")

    rotulos = sorted({linha["rotulo"] for linha in linhas})
    indice_rotulo = {rotulo: indice for indice, rotulo in enumerate(rotulos)}
    caminhos = [Path(linha["caminho_landmarks"]) for linha in linhas]
    ausentes = [caminho for caminho in caminhos if not caminho.is_file()]
    if ausentes:
        raise FileNotFoundError(f"Landmark processado não encontrado: {ausentes[0]}")

    amostras = [np.load(caminho).astype(np.float32) for caminho in caminhos]
    formato = amostras[0].shape
    if any(amostra.shape != formato for amostra in amostras):
        raise ValueError("Todas as sequências precisam ter o mesmo formato")

    alvos = np.asarray([indice_rotulo[linha["rotulo"]] for linha in linhas])
    return np.stack(amostras), alvos, rotulos


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--entrada", type=Path, default=DIR_VLIBRASIL_PROCESSADO)
    analisador.add_argument("--saida", type=Path, default=DIR_MODELOS)
    analisador.add_argument("--epocas", type=int, default=50)
    analisador.add_argument("--tamanho-lote", type=int, default=16)
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    caracteristicas, alvos, rotulos = carregar_amostras(argumentos.entrada.resolve())
    if len(rotulos) < 2:
        raise ValueError("São necessárias pelo menos duas classes para treinar")

    contagens = np.bincount(alvos, minlength=len(rotulos))
    if np.any(contagens < 2):
        insuficientes = [rotulos[i] for i, total in enumerate(contagens) if total < 2]
        raise ValueError(
            "Cada classe precisa de pelo menos duas amostras. Verifique: "
            + ", ".join(insuficientes)
        )

    total_validacao = max(len(rotulos), math.ceil(len(alvos) * 0.2))
    if total_validacao > len(alvos) - len(rotulos):
        raise ValueError("Não há amostras suficientes para treino e validação estratificados")

    treino_x, validacao_x, treino_y, validacao_y = train_test_split(
        caracteristicas,
        alvos,
        test_size=total_validacao,
        random_state=42,
        stratify=alvos,
    )
    modelo = construir_modelo_gru(
        caracteristicas.shape[1], caracteristicas.shape[2], len(rotulos)
    )
    modelo.summary()
    modelo.fit(
        treino_x,
        treino_y,
        validation_data=(validacao_x, validacao_y),
        epochs=argumentos.epocas,
        batch_size=argumentos.tamanho_lote,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
    )

    argumentos.saida.mkdir(parents=True, exist_ok=True)
    modelo.save(argumentos.saida / "modelo_gru.keras")
    (argumentos.saida / "rotulos.json").write_text(
        json.dumps(rotulos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Modelo e rótulos salvos em {argumentos.saida.resolve()}")


if __name__ == "__main__":
    main()
