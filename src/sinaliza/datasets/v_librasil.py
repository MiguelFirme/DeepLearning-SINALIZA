"""Adaptador do V-LIBRASIL, dataset principal do SINALIZA."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from sinaliza.datasets.amostras import AmostraVideo


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
LABEL_COLUMNS = ("class", "label", "sign", "sinal", "word", "gloss")
SIGNER_COLUMNS = ("user_id", "signer_id", "articulator", "articulador", "user")
VIDEO_COLUMNS = ("video_name", "filename", "file", "video", "video_id")
FILENAME_PATTERN = re.compile(
    r"^(?P<label>.+)_(?P<signer>articulador[ _-]*\d+)$", re.IGNORECASE
)


def _first_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    normalized = {key.strip().casefold(): (value or "").strip() for key, value in row.items()}
    return next((normalized[name] for name in names if normalized.get(name)), "")


def _video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )


def _ler_anotacoes(root: Path, annotation_path: Path) -> list[AmostraVideo]:
    videos = _video_files(root)
    by_name = {path.name.casefold(): path for path in videos}
    by_stem = {path.stem.casefold(): path for path in videos}
    samples: list[AmostraVideo] = []

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = _first_value(row, LABEL_COLUMNS)
            signer = _first_value(row, SIGNER_COLUMNS) or "desconhecido"
            video_reference = _first_value(row, VIDEO_COLUMNS)
            path = by_name.get(Path(video_reference).name.casefold())
            if path is None:
                path = by_stem.get(Path(video_reference).stem.casefold())
            if label and path is not None:
                samples.append(
                    AmostraVideo(rotulo=label, articulador_id=signer, caminho=path)
                )
    return samples


def _ler_nomes_arquivos(root: Path) -> list[AmostraVideo]:
    samples: list[AmostraVideo] = []
    for path in _video_files(root):
        match = FILENAME_PATTERN.match(path.stem)
        if match:
            samples.append(
                AmostraVideo(
                    rotulo=match.group("label").replace("_", " ").strip(),
                    articulador_id=match.group("signer").replace("_", " ").strip(),
                    caminho=path,
                )
            )
    return samples


def descobrir_amostras(root: Path) -> list[AmostraVideo]:
    """Cataloga vídeos usando anotações ou a convenção de nomes do V-LIBRASIL."""
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório do V-LIBRASIL não encontrado: {root}")

    annotations = sorted(root.rglob("annotations.csv"), key=lambda path: len(path.parts))
    if annotations:
        samples = _ler_anotacoes(root, annotations[0])
        if samples:
            return samples

    samples = _ler_nomes_arquivos(root)
    if not samples:
        raise ValueError(
            "Não foi possível identificar amostras do V-LIBRASIL. Era esperado um "
            "annotations.csv ou vídeos com nomes como <sinal>_Articulador1.mp4."
        )
    return samples


def gravar_manifesto(samples: list[AmostraVideo], destination: Path) -> None:
    """Grava o catálogo normalizado do V-LIBRASIL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["dataset", "rotulo", "articulador_id", "origem"]
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "dataset": "v_librasil",
                    "rotulo": sample.rotulo,
                    "articulador_id": sample.articulador_id,
                    "origem": str(sample.caminho.resolve()),
                }
            )
