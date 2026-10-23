from pathlib import Path

import pytest

from ml.data.folder_signer import build_manifest
from scripts.extract_landmarks import collect_tasks


def test_folder_signer_manifest_and_class_filter(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "landmarks"
    for label in ("Água", "Beber"):
        (raw / label).mkdir(parents=True)
        (out / label).mkdir(parents=True)
        (raw / label / f"{label}Sinalizador01.mp4").touch()
        (out / label / f"{label}Sinalizador01.npz").touch()

    tasks = collect_tasks(raw, out, {"Água"})
    assert len(tasks) == 1
    assert "Água" in tasks[0][0]
    samples = build_manifest(out, raw)
    assert {(s["label"], s["signer"]) for s in samples} == {
        ("Água", "Sinalizador01"), ("Beber", "Sinalizador01")
    }


def test_folder_signer_rejects_unidentifiable_name(tmp_path: Path):
    (tmp_path / "Água").mkdir()
    (tmp_path / "Água" / "video.npz").touch()
    with pytest.raises(ValueError, match="não identifica"):
        build_manifest(tmp_path)
