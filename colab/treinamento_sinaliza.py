# -*- coding: utf-8 -*-
"""Retreina a BiLSTM criada no notebook original do SINALIZA.

O arquivo exportado pelo Colab continha células exploratórias repetidas e algumas
referências ausentes (``process_video``, ``X_val`` e ``y_val``). Esta versão
preserva a arquitetura e os hiperparâmetros do notebook, mas reutiliza os
landmarks que o projeto já extraiu dos vídeos.

Notebook original:
https://colab.research.google.com/drive/1RkP9dUl8QxtIt3pQSsBDSfDJTTc3j5cQ
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
MANIFESTO = RAIZ_PROJETO / "datasets" / "processados" / "v_librasil" / "manifesto.csv"
DIR_SAIDA = RAIZ_PROJETO / "models" / "treinamento_colega"

QUADROS = 30
CARACTERISTICAS_POSE_ORIGINAIS = 33 * 4
CARACTERISTICAS_FACE = 468 * 3
CARACTERISTICAS_MAOS = 2 * 21 * 3
CARACTERISTICAS_ESPERADAS = 33 * 3 + CARACTERISTICAS_MAOS  # 225


def redimensionar_sequencia(sequencia: np.ndarray, comprimento: int) -> np.ndarray:
    """Interpola uma sequência para a quantidade de quadros usada no notebook."""
    if sequencia.ndim != 2 or len(sequencia) == 0:
        raise ValueError("A sequência de landmarks precisa ter formato (quadros, dados)")
    if len(sequencia) == comprimento:
        return sequencia.astype(np.float32)

    origem = np.arange(len(sequencia), dtype=np.float32)
    destino = np.linspace(0, len(sequencia) - 1, comprimento, dtype=np.float32)
    resultado = np.empty((comprimento, sequencia.shape[1]), dtype=np.float32)
    for coluna in range(sequencia.shape[1]):
        resultado[:, coluna] = np.interp(destino, origem, sequencia[:, coluna])
    return resultado


def converter_para_formato_do_notebook(sequencia: np.ndarray) -> np.ndarray:
    """Converte (quadros, 1662) em pose + mãos no formato (30, 225)."""
    minimo = (
        CARACTERISTICAS_POSE_ORIGINAIS
        + CARACTERISTICAS_FACE
        + CARACTERISTICAS_MAOS
    )
    if sequencia.ndim != 2 or sequencia.shape[1] != minimo:
        raise ValueError(
            f"Formato inesperado {sequencia.shape}; era esperado (quadros, {minimo})"
        )

    pose = sequencia[:, :CARACTERISTICAS_POSE_ORIGINAIS]
    pose_xyz = pose.reshape(len(sequencia), 33, 4)[:, :, :3].reshape(len(sequencia), -1)
    inicio_maos = CARACTERISTICAS_POSE_ORIGINAIS + CARACTERISTICAS_FACE
    maos = sequencia[:, inicio_maos : inicio_maos + CARACTERISTICAS_MAOS]
    dados = np.concatenate((pose_xyz, maos), axis=1)
    return redimensionar_sequencia(dados, QUADROS)


def carregar_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega o manifesto e monta X/y com os landmarks já processados."""
    if not MANIFESTO.is_file():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {MANIFESTO}\n"
            "Execute primeiro: python scripts/processar_vlibrasil.py"
        )

    with MANIFESTO.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise ValueError("O manifesto processado está vazio")

    classes = sorted({linha["rotulo"] for linha in linhas})
    class_to_idx = {nome: indice for indice, nome in enumerate(classes)}
    x: list[np.ndarray] = []
    y: list[int] = []

    print(f"Total de vídeos no manifesto: {len(linhas)}")
    print(f"Total de palavras/classes únicas: {len(classes)}")
    print("\nCarregando landmarks no formato do notebook (30, 225)...")

    for indice, linha in enumerate(linhas, start=1):
        caminho = Path(linha["caminho_landmarks"])
        if not caminho.is_file():
            raise FileNotFoundError(f"Landmark não encontrado: {caminho}")
        sequencia = np.load(caminho, allow_pickle=False)
        x.append(converter_para_formato_do_notebook(sequencia))
        y.append(class_to_idx[linha["rotulo"]])
        if indice % 250 == 0 or indice == len(linhas):
            print(f"Carregados {indice}/{len(linhas)} vídeos...")

    caracteristicas = np.stack(x).astype(np.float32)
    rotulos = np.asarray(y, dtype=np.int32)
    if caracteristicas.shape[1:] != (QUADROS, CARACTERISTICAS_ESPERADAS):
        raise ValueError(f"Formato final inesperado: {caracteristicas.shape}")

    print("\n--- Carregamento concluído ---")
    print("Formato de X (Vídeos, Frames, Keypoints):", caracteristicas.shape)
    print("Formato de y:", rotulos.shape)
    return caracteristicas, rotulos, classes


def salvar_avaliacao(
    history: object,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> None:
    """Salva as mesmas métricas e visualizações produzidas no notebook."""
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(classes))
    acc = accuracy_score(y_val, y_pred)
    relatorio_texto = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
    )
    relatorio_json = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    print("--- RESULTADOS GERAIS ---")
    print(f"Acurácia Geral do Modelo: {acc * 100:.2f}%\n")
    print("--- RELATÓRIO DE CLASSIFICAÇÃO (Precision, Recall, F1-Score) ---")
    print(relatorio_texto)

    (DIR_SAIDA / "metricas.json").write_text(
        json.dumps(
            {"acuracia": float(acc), "relatorio_classificacao": relatorio_json},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (DIR_SAIDA / "historico.json").write_text(
        json.dumps(
            {
                chave: [float(valor) for valor in valores]
                for chave, valores in history.history.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_val, y_pred, labels=labels)
    np.save(DIR_SAIDA / "matriz_confusao.npy", cm)
    mostrar_rotulos = len(classes) <= 40
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=mostrar_rotulos,
        fmt="d",
        cmap="Blues",
        xticklabels=classes if mostrar_rotulos else False,
        yticklabels=classes if mostrar_rotulos else False,
    )
    plt.title("Matriz de Confusão - Sinaliza Bi-LSTM")
    plt.xlabel("Classe Predita pela IA")
    plt.ylabel("Classe Real (Gabarito)")
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "matriz_confusao.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history.history["accuracy"], label="Treino", linewidth=2)
    ax[0].plot(history.history["val_accuracy"], label="Validação", linewidth=2)
    ax[0].set_title("Evolução da Acurácia")
    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("Acurácia")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history.history["loss"], label="Treino", linewidth=2)
    ax[1].plot(history.history["val_loss"], label="Validação", linewidth=2)
    ax[1].set_title("Evolução da Perda (Loss)")
    ax[1].set_xlabel("Época")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / "curvas_aprendizado.png", dpi=160)
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    x, y, classes = carregar_dataset()

    # Divisão mantida como no código original da colega.
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    print("Shape X_train:", x_train.shape)
    print("Shape X_val:", x_val.shape)

    # Arquitetura mantida como no notebook original.
    num_classes = len(classes)
    model = Sequential(
        [
            Input(shape=(x.shape[1], x.shape[2])),
            Bidirectional(
                LSTM(64, return_sequences=True),
            ),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=8,
    )

    y_pred_probs = model.predict(x_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    salvar_avaliacao(history, y_val, y_pred, classes)

    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    model_filename = DIR_SAIDA / "sinaliza_bilstm_model.h5"
    model.save(model_filename)
    (DIR_SAIDA / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Modelo salvo como '{model_filename}'!")

    # Mantém o download automático quando o mesmo arquivo for executado no Colab.
    if "google.colab" in sys.modules:
        from google.colab import files

        files.download(str(model_filename))


if __name__ == "__main__":
    main()
