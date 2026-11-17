"""Conjunto ampliado: novas classes, contagens diferentes e identidade isolada."""
import json

import pytest

from scripts.build_mixed_group_dataset import assemble
from scripts.build_mixed_23_dataset import MINDS_CLASSES, PROTOCOL, GROUP_CLASSES


def test_inclui_classes_novas_e_nao_inventa_escola04(tmp_path):
    minds, group, raw = tmp_path / "minds", tmp_path / "group", tmp_path / "raw"
    classes = GROUP_CLASSES | {"Diretoria", "Dor", "Dormir", "Dúvida", "Entristecer", "Errado", "Escola", "Escrever", "Estudar",
                               "Eu", "Feliz", "Meu", "Mãe", "Pai", "Perguntar", "Professor", "Querer", "Querer-não"}
    for origin, source, labels in (("minds", minds, MINDS_CLASSES), ("novas", group, classes)):
        source.mkdir()
        rows = []
        for label in sorted(labels):
            for people in PROTOCOL[origin].values():
                for person in sorted(people):
                    if label == "Escola" and person == "Sinalizador04":
                        continue
                    for take in range(5 if origin == "minds" else 1):
                        path = f"{label}/{label}{person}_{take}.npz"
                        (source / path).parent.mkdir(exist_ok=True)
                        (source / path).touch()
                        if origin == "novas":
                            video = raw / path.replace(".npz", ".mp4")
                            video.parent.mkdir(parents=True, exist_ok=True)
                            video.touch()
                        rows.append({"path": path, "label": label, "signer": person})
        (source / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    output, processed = tmp_path / "mixed", tmp_path / "processed"
    report = assemble(minds, group, raw, output, processed)
    assert report["num_samples"] == 715 and report["num_classes"] == 41
    assert report["split_sizes"] == {"train": 529, "val": 93, "test": 93}
    assert report["class_counts_by_split"]["train"]["Escola"] == 4
    splits = json.loads((processed / "splits.json").read_text())
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    people = {key: {manifest[i]["person"] for i in indices} for key, indices in splits.items()}
    assert not people["train"] & people["test"] and not people["val"] & people["test"]
    assert assemble(minds, group, raw, output, processed) == report
    # Não permite usar um cache antigo que deixe um MP4 recém-adicionado de fora.
    (raw / "Errado" / "extra.mp4").touch()
    with pytest.raises(ValueError, match="não corresponde"):
        assemble(minds, group, raw, output, processed)
    (raw / "Errado" / "extra.mp4").unlink()
    (group / "manifest.json").write_text(json.dumps(rows + [rows[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        assemble(minds, group, raw, output, processed)
    rows[0]["signer"] = "Sinalizador08"
    (group / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Pessoa fora"):
        assemble(minds, group, raw, output, processed)
