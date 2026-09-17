"""Extração paralela das sequências do V-LIBRASIL."""

from __future__ import annotations

import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sinaliza.datasets.amostras import AmostraVideo
from sinaliza.landmarks import FEATURE_DIM, criar_holistic, extrair_video
from sinaliza.sequencias import redimensionar_sequencia


_HOLISTIC_WORKER: object | None = None


def _nome_caracteristica(amostra: AmostraVideo) -> str:
    identidade = str(amostra.caminho.resolve()).encode("utf-8")
    return f"{hashlib.sha1(identidade).hexdigest()[:16]}.npy"


def _linha_manifesto(
    amostra: AmostraVideo, destino: Path, sequencia: np.ndarray
) -> dict[str, str | int]:
    return {
        "dataset": "v_librasil",
        "rotulo": amostra.rotulo,
        "articulador_id": amostra.articulador_id,
        "origem": str(amostra.caminho.resolve()),
        "caminho_landmarks": str(destino.resolve()),
        "quadros": sequencia.shape[0],
        "caracteristicas": sequencia.shape[1],
    }


def _carregar_processado(
    amostra: AmostraVideo, destino: Path, comprimento_sequencia: int
) -> dict[str, str | int] | None:
    if not destino.is_file():
        return None
    try:
        sequencia = np.load(destino, mmap_mode="r", allow_pickle=False)
        if sequencia.shape != (comprimento_sequencia, FEATURE_DIM):
            return None
        return _linha_manifesto(amostra, destino, sequencia)
    except (OSError, ValueError):
        return None


def _inicializar_worker() -> None:
    global _HOLISTIC_WORKER
    _HOLISTIC_WORKER = criar_holistic()


def _processar_amostra(
    indice: int,
    amostra: AmostraVideo,
    destino: Path,
    comprimento_sequencia: int,
    holistic: object | None = None,
) -> tuple[int, dict[str, str | int] | None, str | None]:
    detector = holistic if holistic is not None else _HOLISTIC_WORKER
    temporario = destino.with_name(f".{destino.stem}.{os.getpid()}.tmp.npy")
    try:
        resetar = getattr(detector, "reset", None)
        if callable(resetar):
            resetar()
        sequencia = redimensionar_sequencia(
            extrair_video(amostra.caminho, detector), comprimento_sequencia
        )
        np.save(temporario, sequencia)
        temporario.replace(destino)
        return indice, _linha_manifesto(amostra, destino, sequencia), None
    except (OSError, RuntimeError, ValueError) as error:
        return indice, None, str(error)
    finally:
        temporario.unlink(missing_ok=True)


def processar_vlibrasil(
    amostras: list[AmostraVideo],
    diretorio_saida: Path,
    comprimento_sequencia: int,
    processos: int | None = None,
) -> None:
    """Extrai landmarks em paralelo e retoma resultados válidos existentes."""
    if processos is not None and processos < 1:
        raise ValueError("A quantidade de processos precisa ser maior que zero")

    diretorio_landmarks = diretorio_saida / "landmarks"
    diretorio_landmarks.mkdir(parents=True, exist_ok=True)
    linhas: dict[int, dict[str, str | int]] = {}
    pendentes: list[tuple[int, AmostraVideo, Path]] = []
    total_falhas = 0

    for indice, amostra in enumerate(amostras, start=1):
        destino = diretorio_landmarks / _nome_caracteristica(amostra)
        linha = _carregar_processado(amostra, destino, comprimento_sequencia)
        if linha is None:
            pendentes.append((indice, amostra, destino))
        else:
            linhas[indice] = linha

    if linhas:
        print(f"Reaproveitando {len(linhas)} vídeo(s) já processado(s).")

    total_processos = processos or min(4, max(1, (os.cpu_count() or 2) // 2))
    total_processos = min(total_processos, max(1, len(pendentes)))
    if pendentes:
        print(
            f"Processando {len(pendentes)} vídeo(s) com {total_processos} "
            "processo(s) em paralelo."
        )

    def registrar(
        resultado: tuple[int, dict[str, str | int] | None, str | None]
    ) -> None:
        nonlocal total_falhas
        indice, linha, erro = resultado
        amostra = amostras[indice - 1]
        if linha is None:
            total_falhas += 1
            print(f"[{indice}/{len(amostras)}] AVISO: {amostra.caminho.name}: {erro}")
        else:
            linhas[indice] = linha
            print(f"[{len(linhas)}/{len(amostras)}] concluído: {amostra.caminho.name}")

    if total_processos == 1 and pendentes:
        with criar_holistic() as detector:
            for indice, amostra, destino in pendentes:
                registrar(
                    _processar_amostra(
                        indice, amostra, destino, comprimento_sequencia, detector
                    )
                )
    elif pendentes:
        with ProcessPoolExecutor(
            max_workers=total_processos, initializer=_inicializar_worker
        ) as executor:
            futuros = [
                executor.submit(
                    _processar_amostra,
                    indice,
                    amostra,
                    destino,
                    comprimento_sequencia,
                )
                for indice, amostra, destino in pendentes
            ]
            for futuro in as_completed(futuros):
                registrar(futuro.result())

    manifesto = diretorio_saida / "manifesto.csv"
    manifesto_temporario = manifesto.with_suffix(".tmp")
    with manifesto_temporario.open("w", newline="", encoding="utf-8") as file:
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
        escritor.writerows(linhas[indice] for indice in sorted(linhas))
    manifesto_temporario.replace(manifesto)

    print(f"Processamento concluído: {len(linhas)} sucesso(s), {total_falhas} falha(s).")
