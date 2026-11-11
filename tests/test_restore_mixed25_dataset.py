"""Restaurar índices não pode fingir que landmarks ausentes foram recuperados."""
import json
from pathlib import Path

import pytest

from ml.training.pipeline_v2 import check_dataset_files
from scripts.restore_mixed25_dataset import restore_dataset


ARCHIVE = Path(__file__).resolve().parents[1] / "Modelos salvos/transformer_mixed25_full/data_split"


def test_diagnostica_ausentes_sem_escrever_indices(tmp_path):
    root, processed = tmp_path / "landmarks", tmp_path / "processed"
    report = restore_dataset(ARCHIVE, root, processed)
    assert report["found"] == 0 and report["missing"] == 670
    assert not report["restored"]
    assert not root.exists() and not processed.exists()
    with pytest.raises(FileNotFoundError, match="restore_mixed25_dataset"):
        check_dataset_files(root / "mixed_25", processed)


def test_restaura_indices_sem_sobrescrever_ou_criar_landmarks(tmp_path):
    # Só é verificada existência aqui. A validação de arrays pertence ao dataset.
    root, processed = tmp_path / "landmarks", tmp_path / "processed"
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest:
        path = root / "mixed_25" / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    report = restore_dataset(ARCHIVE, root, processed)
    assert report["restored"] and report["found"] == 670
    assert (processed / "splits.json").read_bytes() == (ARCHIVE / "splits.json").read_bytes()
    check_dataset_files(root / "mixed_25", processed)
    (processed / "splits.json").write_text("{}")
    with pytest.raises(ValueError, match="sem sobrescrever"):
        restore_dataset(ARCHIVE, root, processed)
