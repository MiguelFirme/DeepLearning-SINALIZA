"""Vocabulário de 23 classes e holdout por pessoa após as exclusões."""
import json

import pytest

from scripts.build_mixed_23_dataset import assemble_mixed23, MINDS_CLASSES, GROUP_CLASSES, PROTOCOL


def test_monta_590_amostras_e_preserva_pessoas(tmp_path):
    sources = {}
    for origin, classes in (("minds", MINDS_CLASSES), ("novas", GROUP_CLASSES)):
        source = tmp_path / origin
        source.mkdir()
        rows = []
        for label in sorted(classes):
            for people in PROTOCOL[origin].values():
                for person in sorted(people):
                    for take in range(5 if origin == "minds" else 1):
                        name = f"{label}_{person}_{take}.npz"
                        (source / name).touch()
                        rows.append({"path": name, "label": label, "signer": person})
        (source / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
        sources[origin] = source
    output, processed = tmp_path / "mixed_23", tmp_path / "processed"
    report = assemble_mixed23(sources["minds"], sources["novas"], output, processed)
    assert report["num_samples"] == 590 and report["num_classes"] == 23
    assert report["split_sizes"] == {"train": 440, "val": 75, "test": 75}
    labels = json.loads((processed / "label_map.json").read_text(encoding="utf-8"))
    assert "Cinco" not in labels and "Espelho" not in labels and "Água" in labels
    assert assemble_mixed23(sources["minds"], sources["novas"], output, processed) == report
    (sources["novas"] / rows[0]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        assemble_mixed23(sources["minds"], sources["novas"], output, processed)
