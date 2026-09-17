from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sinaliza.datasets.amostras import AmostraVideo
from sinaliza.landmarks import FEATURE_DIM
from sinaliza.processamento import _nome_caracteristica, processar_vlibrasil


def test_reaproveita_landmarks_validos(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "Oi_Articulador1.mp4"
    video.touch()
    amostra = AmostraVideo("Oi", "Articulador1", video)
    saida = tmp_path / "processados"
    landmarks = saida / "landmarks"
    landmarks.mkdir(parents=True)
    destino = landmarks / _nome_caracteristica(amostra)
    np.save(destino, np.zeros((60, FEATURE_DIM), dtype=np.float32))

    def falhar_se_reprocessar(*args, **kwargs):
        raise AssertionError("Um landmark válido não deveria ser reprocessado")

    monkeypatch.setattr("sinaliza.processamento.criar_holistic", falhar_se_reprocessar)
    processar_vlibrasil([amostra], saida, 60, processos=1)

    with (saida / "manifesto.csv").open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert len(linhas) == 1
    assert linhas[0]["rotulo"] == "Oi"
    assert Path(linhas[0]["caminho_landmarks"]) == destino.resolve()


def test_rejeita_quantidade_de_processos_invalida(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        processar_vlibrasil([], tmp_path, 60, processos=0)
