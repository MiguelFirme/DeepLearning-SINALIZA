"""Verifica a seleção em tempo de execução e a proteção de classes úteis."""
import csv
import json
import sys

import pytest

from ml.data.vocabulary import (classify_label, read_annotations, read_curated_vocabulary,
                                select_essential_categories, select_vocabulary)
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


def test_school_needs_vocabulary_has_100_covered_classes():
    selected = read_curated_vocabulary("configs/vocabulary_school_needs_100.json")
    annotations = read_annotations("data/annotations.csv")
    assert len(selected) == 100
    assert set(selected) <= set(annotations)
    assert all(set(signers) == {"Articulador1", "Articulador2", "Articulador3"}
               for label, signers in annotations.items() if label in selected)
    assert {"Onde", "Imprimir", "Dúvida (incerteza)", "Banheiro", "Fome"} <= set(selected)
    assert "Abacaxi" not in selected


def test_curated_vocabulary_rejects_duplicate(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"pedidos": ["Banheiro"], "locais": ["Banheiro"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicada"):
        read_curated_vocabulary(path)


def test_build_dataset_uses_exact_curated_classes(tmp_path, monkeypatch):
    data_dir = tmp_path / "landmarks"
    data_dir.mkdir()
    annotation_path = tmp_path / "annotations.csv"
    rows = []
    samples = []
    for label in ("Banheiro", "Ajudar", "Abacaxi"):
        for signer in ("Articulador1", "Articulador2", "Articulador3"):
            rows.append({"video_id": f"{label}_{signer}.mp4", "class": label, "user_id": signer})
            samples.append({"path": f"{label}_{signer}.npz", "label": label, "signer": signer})
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "manifest.json").write_text(json.dumps(samples), encoding="utf-8")
    class_list = tmp_path / "classes.json"
    class_list.write_text(json.dumps({"locais": ["Banheiro"]}), encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--landmarks", str(data_dir), "--output", str(output),
        "--annotations", str(annotation_path), "--class-list", str(class_list),
        "--max-classes", "1", "--strategy", "signer_holdout", "--test-signer", "Articulador3",
    ])
    main()
    labels = json.loads((output / "label_map.json").read_text(encoding="utf-8"))
    report = json.loads((output / "vocabulary_report.json").read_text(encoding="utf-8"))
    assert set(labels) == {"Banheiro"}
    assert report["included"] == {"Banheiro": "curado:locais"}
    assert set(report["excluded"]) == {"Ajudar", "Abacaxi"}


def test_school_category_selects_only_present_labels():
    annotations = read_annotations("data/annotations.csv")
    selected = select_essential_categories(set(annotations), ["escola"])
    assert len(selected) == 23
    assert "Secretária" in selected
    assert "Banheiro" not in selected
    assert "Diretoria" not in selected
    assert set(selected.values()) == {"categoria:escola"}
    with pytest.raises(ValueError, match="desconhecidas"):
        select_essential_categories(set(annotations), ["inexistente"])


def test_build_dataset_uses_category_selection(tmp_path, monkeypatch):
    data_dir = tmp_path / "landmarks"
    data_dir.mkdir()
    annotation_path = tmp_path / "annotations.csv"
    rows = []
    samples = []
    for label in ("Biblioteca", "Ajudar", "Abacaxi"):
        for signer in ("Articulador1", "Articulador2", "Articulador3"):
            rows.append({"video_id": f"{label}_{signer}.mp4", "class": label, "user_id": signer})
            samples.append({"path": f"{label}_{signer}.npz", "label": label, "signer": signer})
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "manifest.json").write_text(json.dumps(samples), encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--landmarks", str(data_dir), "--output", str(output),
        "--annotations", str(annotation_path), "--essential-categories", "escola",
        "--max-classes", "1", "--strategy", "signer_holdout", "--test-signer", "Articulador3",
    ])
    main()
    labels = json.loads((output / "label_map.json").read_text(encoding="utf-8"))
    report = json.loads((output / "vocabulary_report.json").read_text(encoding="utf-8"))
    assert set(labels) == {"Biblioteca"}
    assert report["included"] == {"Biblioteca": "categoria:escola"}
