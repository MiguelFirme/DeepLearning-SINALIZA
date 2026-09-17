"""Treina e avalia um modelo temporal com os landmarks do V-LIBRASIL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sinaliza.config import DIR_MODELOS, DIR_VLIBRASIL_PROCESSADO  # noqa: E402
from sinaliza.modelo import construir_modelo_bilstm, construir_modelo_gru  # noqa: E402


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


def salvar_avaliacao(
    saida: Path,
    historico: keras.callbacks.History,
    modelo: keras.Model,
    validacao_x: np.ndarray,
    validacao_y: np.ndarray,
    rotulos: list[str],
) -> None:
    """Salva as métricas e os gráficos que antes existiam apenas no notebook."""
    diretorio = saida / "avaliacao"
    diretorio.mkdir(parents=True, exist_ok=True)

    probabilidades = modelo.predict(validacao_x, verbose=0)
    previstos = np.argmax(probabilidades, axis=1)
    indices = np.arange(len(rotulos))
    relatorio = classification_report(
        validacao_y,
        previstos,
        labels=indices,
        target_names=rotulos,
        output_dict=True,
        zero_division=0,
    )
    metricas = {
        "acuracia": float(accuracy_score(validacao_y, previstos)),
        "relatorio_classificacao": relatorio,
    }
    (diretorio / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    historico_serializavel = {
        chave: [float(valor) for valor in valores]
        for chave, valores in historico.history.items()
    }
    (diretorio / "historico.json").write_text(
        json.dumps(historico_serializavel, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figura, eixos = plt.subplots(1, 2, figsize=(14, 5))
    eixos[0].plot(historico.history["accuracy"], label="Treino")
    eixos[0].plot(historico.history["val_accuracy"], label="Validação")
    eixos[0].set(title="Evolução da acurácia", xlabel="Época", ylabel="Acurácia")
    eixos[0].legend()
    eixos[0].grid(True)
    eixos[1].plot(historico.history["loss"], label="Treino")
    eixos[1].plot(historico.history["val_loss"], label="Validação")
    eixos[1].set(title="Evolução da perda", xlabel="Época", ylabel="Perda")
    eixos[1].legend()
    eixos[1].grid(True)
    figura.tight_layout()
    figura.savefig(diretorio / "curvas_aprendizado.png", dpi=160)
    plt.close(figura)

    matriz = confusion_matrix(validacao_y, previstos, labels=indices)
    np.save(diretorio / "matriz_confusao.npy", matriz)
    figura, eixo = plt.subplots(figsize=(10, 8))
    imagem = eixo.imshow(matriz, interpolation="nearest", cmap="Blues")
    eixo.set(title="Matriz de confusão - SINALIZA", xlabel="Predito", ylabel="Real")
    if len(rotulos) <= 40:
        eixo.set_xticks(indices, rotulos, rotation=90, fontsize=7)
        eixo.set_yticks(indices, rotulos, fontsize=7)
    else:
        eixo.set_xticks([])
        eixo.set_yticks([])
    figura.colorbar(imagem, ax=eixo)
    figura.tight_layout()
    figura.savefig(diretorio / "matriz_confusao.png", dpi=160)
    plt.close(figura)


def ler_argumentos() -> argparse.Namespace:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--entrada", type=Path, default=DIR_VLIBRASIL_PROCESSADO)
    analisador.add_argument("--saida", type=Path, default=DIR_MODELOS)
    analisador.add_argument("--epocas", type=int, default=30)
    analisador.add_argument("--tamanho-lote", type=int, default=8)
    analisador.add_argument(
        "--arquitetura", choices=("bilstm", "gru"), default="bilstm"
    )
    analisador.add_argument("--semente", type=int, default=42)
    return analisador.parse_args()


def main() -> None:
    argumentos = ler_argumentos()
    keras.utils.set_random_seed(argumentos.semente)
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
        random_state=argumentos.semente,
        stratify=alvos,
    )
    construtores = {
        "bilstm": construir_modelo_bilstm,
        "gru": construir_modelo_gru,
    }
    modelo = construtores[argumentos.arquitetura](
        caracteristicas.shape[1], caracteristicas.shape[2], len(rotulos)
    )
    modelo.summary()
    historico = modelo.fit(
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

    saida = argumentos.saida.resolve()
    saida.mkdir(parents=True, exist_ok=True)
    nome_modelo = f"modelo_{argumentos.arquitetura}.keras"
    modelo.save(saida / nome_modelo)
    (saida / "rotulos.json").write_text(
        json.dumps(rotulos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (saida / "modelo_atual.json").write_text(
        json.dumps(
            {"arquivo": nome_modelo, "arquitetura": argumentos.arquitetura},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    salvar_avaliacao(saida, historico, modelo, validacao_x, validacao_y, rotulos)
    print(f"Modelo, rótulos e avaliação salvos em {saida}")


if __name__ == "__main__":
    main()
