"""Rótulos e proveniência específicos do MINDS-Libras."""
import csv
import json

import pytest

from ml.data.minds_libras import (build_manifest, parse_video_name,
                                  write_manifest_and_annotations)
from scripts.download_minds_libras import download_file, select_files


def test_filename_encodes_class_signer_and_take():
    fields = parse_video_name("08BanheiroSinalizador12-5.mp4")
    assert fields == {"label": "Banheiro", "class_number": "08",
                      "signer": "Sinalizador12", "take": 5}
    with pytest.raises(ValueError, match="inválido"):
        parse_video_name("video_qualquer.mp4")


def test_manifest_and_annotations_keep_video_and_person(tmp_path):
    landmarks = tmp_path / "landmarks"
    landmarks.mkdir()
    for name in ("08BanheiroSinalizador01-1.npz", "08BanheiroSinalizador01-2.npz",
                 "08BanheiroSinalizador12-1.npz"):
        (landmarks / name).touch()
    samples = build_manifest(landmarks)
    write_manifest_and_annotations(samples, landmarks / "manifest.json",
                                   tmp_path / "annotations.csv")
    assert len(samples) == 3
    assert {s["signer"] for s in samples} == {"Sinalizador01", "Sinalizador12"}
    assert {s["video_id"] for s in samples} == {
        "08BanheiroSinalizador01-1.mp4", "08BanheiroSinalizador01-2.mp4",
        "08BanheiroSinalizador12-1.mp4"}
    assert json.loads((landmarks / "manifest.json").read_text(encoding="utf-8")) == samples
    with (tmp_path / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["class"] == "Banheiro"


def test_select_files_fails_on_missing_class():
    files = [{"name": "08BanheiroSinalizador01-1.mp4", "bytes": 5},
             {"name": "08BanheiroSinalizador12-1.mp4", "bytes": 6}]
    assert len(select_files(files, {"Banheiro"})) == 2
    with pytest.raises(ValueError, match="ausentes"):
        select_files(files, {"Aluno"})


def test_download_releases_video_only_after_size_matches(tmp_path, monkeypatch):
    file = {"name": "08BanheiroSinalizador01-1.mp4", "bytes": 10}
    part = tmp_path / (file["name"] + ".part")
    part.write_bytes(b"123")
    def finish(command, check):
        assert command[command.index("--continue-at") + 1] == "-"
        assert part.exists()
        part.write_bytes(b"1234567890")
    monkeypatch.setattr("scripts.download_minds_libras.subprocess.run", finish)
    download_file(file, tmp_path)
    assert (tmp_path / file["name"]).read_bytes() == b"1234567890"
    assert not part.exists()
