"""Manifesto para vídeos organizados como Classe/ClasseSinalizadorNN.mp4."""
from __future__ import annotations

import re
from pathlib import Path

from ml.data.minds_libras import write_manifest_and_annotations


def build_manifest(landmarks_dir: Path, videos_dir: Path | None = None) -> list[dict]:
    samples = []
    for npz in sorted(landmarks_dir.rglob("*.npz")):
        relative = npz.relative_to(landmarks_dir)
        if len(relative.parts) != 2:
            raise ValueError(f"Esperado Classe/video.npz: {relative}")
        label = relative.parent.name
        match = re.fullmatch(rf"{re.escape(label)}Sinalizador(\d{{2}})", npz.stem,
                             flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Vídeo não identifica classe e pessoa: {relative}")
        video_relative = relative.with_suffix(".mp4")
        if videos_dir is not None and not (videos_dir / video_relative).is_file():
            raise FileNotFoundError(videos_dir / video_relative)
        samples.append({
            "path": relative.as_posix(),
            "video_id": video_relative.as_posix(),
            "label": label,
            "signer": f"Sinalizador{match.group(1)}",
            "take": 1,
        })
    if not samples:
        raise ValueError(f"Nenhum landmark .npz em {landmarks_dir}")
    seen = set()
    for sample in samples:
        key = (sample["label"], sample["signer"])
        if key in seen:
            raise ValueError(f"Classe/pessoa duplicada: {key}")
        seen.add(key)
    return samples


__all__ = ["build_manifest", "write_manifest_and_annotations"]
