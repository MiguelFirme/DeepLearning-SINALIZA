"""Testes dos splits por articulador."""

import pytest

from ml.data.split import DataSplitter, SplitConfig


def _samples():
    samples = []
    for label in ("A", "B", "C"):
        for signer in ("Articulador1", "Articulador2", "Articulador3"):
            samples.append({"label": label, "signer": signer, "path": f"{label}_{signer}.npz"})
    samples.append({"label": "Incompleta", "signer": "Articulador1", "path": "inc_1.npz"})
    samples.append({"label": "Incompleta", "signer": "Articulador2", "path": "inc_2.npz"})
    return samples


def test_signer_holdout_uses_two_signers_and_reserves_third():
    samples = _samples()
    splitter = DataSplitter(SplitConfig(strategy="group"))
    labels = set(splitter.labels_with_signer_coverage(
        samples, ["Articulador1", "Articulador2", "Articulador3"]
    ))

    result = splitter.split_by_signer(
        samples,
        test_signer="Articulador3",
        train_signers=["Articulador1", "Articulador2"],
        allowed_labels=labels,
    )

    assert labels == {"A", "B", "C"}
    assert len(result["train"]) == 6
    assert result["val"] == []
    assert len(result["test"]) == 3
    assert {samples[i]["signer"] for i in result["train"]} == {
        "Articulador1", "Articulador2"
    }
    assert {samples[i]["signer"] for i in result["test"]} == {"Articulador3"}


def test_signer_holdout_rejects_test_signer_in_training():
    splitter = DataSplitter()
    with pytest.raises(ValueError, match="não pode aparecer no treino"):
        splitter.split_by_signer(
            _samples(),
            test_signer="Articulador3",
            train_signers=["Articulador1", "Articulador3"],
        )


def test_signer_holdout_rejects_unknown_signer():
    splitter = DataSplitter()
    with pytest.raises(ValueError, match="desconhecido"):
        splitter.split_by_signer(_samples(), test_signer="Articulador4")
