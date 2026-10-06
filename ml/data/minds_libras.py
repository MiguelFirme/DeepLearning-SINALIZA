"""Nomes, proveniência e anotações do MINDS-Libras (Kaggle, versão 3)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

FILENAME = re.compile(
    r"^(?P<class_number>\d{2})(?P<label>.+)Sinalizador(?P<signer_number>\d{2})-(?P<take>[1-9]\d*)\.mp4$",
    re.IGNORECASE,
)


def parse_video_name(name: str) -> dict:
    """Falha explicitamente se o nome não codificar classe, pessoa e tomada."""
    match = FILENAME.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Nome MINDS-Libras inválido: {name}")
    fields = match.groupdict()
    return {
        "label": fields["label"],
        "class_number": fields["class_number"],
        "signer": f"Sinalizador{fields['signer_number']}",
        "take": int(fields["take"]),
    }


def build_manifest(landmarks_dir: Path, videos_dir: Path | None = None) -> list[dict]:
    """Gera manifesto apenas de .npz cujo vídeo de origem é identificável."""
    samples = []
    for npz in sorted(landmarks_dir.rglob("*.npz")):
        relative = npz.relative_to(landmarks_dir)
        video_relative = relative.with_suffix(".mp4")
        fields = parse_video_name(video_relative.name)
        if videos_dir is not None and not (videos_dir / video_relative).is_file():
            raise FileNotFoundError(videos_dir / video_relative)
        samples.append({
            "path": relative.as_posix(),
            "video_id": video_relative.as_posix(),
            **fields,
        })
    if not samples:
        raise ValueError(f"Nenhum landmark .npz em {landmarks_dir}")
    # Um identificador de classe deve apontar sempre para a mesma palavra.
    labels_by_number: dict[str, str] = {}
    keys = set()
    for sample in samples:
        number, label = sample["class_number"], sample["label"]
        if number in labels_by_number and labels_by_number[number] != label:
            raise ValueError(f"Classe {number} tem rótulos conflitantes")
        labels_by_number[number] = label
        key = (label, sample["signer"], sample["take"])
        if key in keys:
            raise ValueError(f"Vídeo duplicado: {key}")
        keys.add(key)
    return samples


def write_manifest_and_annotations(samples: list[dict], manifest_path: Path,
                                   annotations_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with annotations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"video_id": sample["video_id"], "class": sample["label"],
                             "user_id": sample["signer"]})
