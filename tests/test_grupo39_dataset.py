"""A remoção de classes mantém as pessoas nos splits originais."""
from copy import deepcopy

from scripts.prepare_grupo39_dataset import filter_indices


def test_filtragem_remapeia_indices_sem_mover_pessoas():
    rows = [{"label": word, "signer": person} for word, person in
            [("Banheiro", "P1"), ("Amarelo", "P1"), ("Vacina", "P2"),
             ("Escola", "P2"), ("Cinco", "P3"), ("Querer", "P3")]]
    splits = {"train": [0, 1], "val": [2, 3], "test": [4, 5]}
    original = deepcopy((rows, splits))
    filtered, indices = filter_indices(rows, splits)
    assert indices == {"train": [0], "val": [1], "test": [2]}
    assert [row["label"] for row in filtered] == ["Banheiro", "Escola", "Querer"]
    for split, old_ids in splits.items():
        assert {filtered[i]["signer"] for i in indices[split]} <= {rows[i]["signer"] for i in old_ids}
    assert (rows, splits) == original
