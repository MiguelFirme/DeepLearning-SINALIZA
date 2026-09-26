"""Verifica a seleção em tempo de execução e a proteção de classes úteis."""
import csv
import json
import sys

import pytest

from ml.data.vocabulary import classify_label, read_annotations, select_vocabulary
from ml.data.dataset import LibrasDataset
from scripts.build_dataset import main


@pytest.mark.parametrize("label", [
    "Ajudar", "Abrir a porta", "Congelar", "Solicitar", "Banheiro",
    "Água", "Dor de cabeça", "Escola", "Obrigado", "Quer", "Ir",
    "Faz", "Vai", "Espere", "Eu vejo", "Use língua de sinais",
    "Com licença", "Emergência", "Documento",
])
def test_protected_classes(label):
    assert classify_label(label) is not None


@pytest.mark.parametrize("label", ["Abacaxi", "Banana", "Borrifador", "Cobertor"])
def test_irrelevant_classes(label):
    assert classify_label(label) is None


def test_build_dataset_filters_csv_before_splitting(tmp_path, monkeypatch):
    data_dir = tmp_path / "landmarks"
    data_dir.mkdir()
    annotation_path = tmp_path / "annotations.csv"
    rows = []
    samples = []
    for label, signers in {
        "Banheiro": ("Articulador1", "Articulador2", "Articulador3"),
        "Ajudar": ("Articulador1", "Articulador2", "Articulador3"),
        "Congelar": ("Articulador1", "Articulador2"),
        "Abacaxi": ("Articulador1", "Articulador2", "Articulador3"),
    }.items():
        for signer in signers:
            rows.append({"video_id": f"{label}_{signer}.mp4", "class": label, "user_id": signer})
            samples.append({"path": f"{label}_{signer}.npz", "label": label, "signer": signer})
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "manifest.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--landmarks", str(data_dir), "--output", str(output),
        "--annotations", str(annotation_path), "--strategy", "signer_holdout",
        "--test-signer", "Articulador3",
    ])
    main()
    labels = json.loads((output / "label_map.json").read_text(encoding="utf-8"))
    report = json.loads((output / "vocabulary_report.json").read_text(encoding="utf-8"))
    splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert set(labels) == {"Banheiro", "Ajudar", "Congelar"}
    assert "Abacaxi" in report["excluded"]
    assert report["included"]["Congelar"] == "verbo"
    assert len(splits["train"]) == 6
    assert len(splits["test"]) == 2


def test_missing_columns_fail(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("class,user_id\nBanheiro,Articulador1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colunas"):
        read_annotations(path)


def test_dataset_never_emits_unselected_label(tmp_path):
    samples = [
        {"path": "a.npz", "label": "Banheiro", "signer": "Articulador1"},
        {"path": "b.npz", "label": "Abacaxi", "signer": "Articulador1"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(samples), encoding="utf-8")
    dataset = LibrasDataset(tmp_path, label_map={"Banheiro": 0})
    assert len(dataset) == 1
    assert dataset.get_labels() == [0]


def test_real_annotations_protect_core_vocabulary():
    annotations = read_annotations("data/annotations.csv")
    selected = select_vocabulary(set(annotations))
    for label in ("Banheiro", "Água", "Escola", "Ajudar", "Ir", "Faz", "Vai"):
        assert label in selected
    for label in ("Abacaxi", "Banana", "Melancia"):
        if label in annotations:
            assert label not in selected
