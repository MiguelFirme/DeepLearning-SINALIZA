#!/usr/bin/env python3
"""Une os landmarks das dez palavras novas sem copiar os arquivos .npz."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.minds_libras import write_manifest_and_annotations


EXPECTED = {"Água", "Ajudar", "Amigo", "Aprender", "Beber",
            "Brincar", "Cansado", "Certo", "Comer", "Diretor"}
SIGNERS = {f"Sinalizador{i:02d}" for i in range(1, 8)}


def build_manifest(sources: list[Path], output: Path) -> list[dict]:
    if len(sources) != 2 or sources[0].resolve() == sources[1].resolve():
        raise ValueError("Informe dois diretórios de origem distintos")
    output = output.resolve()
    if any(source.resolve().parent != output.parent for source in sources):
        raise ValueError("As origens e a saída precisam ser subpastas irmãs")
    if any(source.resolve() == output for source in sources):
        raise ValueError("A saída não pode sobrescrever uma origem")

    samples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        for sample in original:
            label, signer = sample["label"], sample["signer"]
            if label not in EXPECTED or signer not in SIGNERS:
                raise ValueError(f"Classe ou pessoa fora do protocolo: {label}/{signer}")
            key = (label, signer)
            if key in seen:
                raise ValueError(f"Classe/pessoa duplicada: {key}")
            seen.add(key)
            if not (source / sample["path"]).is_file():
                raise FileNotFoundError(source / sample["path"])
            record = dict(sample)
            record["path"] = (Path("..") / source.name / sample["path"]).as_posix()
            record["video_id"] = f"{source.name}/{sample['video_id']}"
            record["origin"] = source.name
            samples.append(record)

    if seen != {(label, signer) for label in EXPECTED for signer in SIGNERS}:
        raise ValueError("Esperados exatamente 70 vídeos: uma amostra por classe/pessoa")
    counts = Counter(sample["label"] for sample in samples)
    if set(counts.values()) != {7}:
        raise ValueError(f"Contagens por classe incorretas: {counts}")
    samples.sort(key=lambda s: (s["label"], s["signer"]))
    output.mkdir(parents=True, exist_ok=True)
    write_manifest_and_annotations(samples, output / "manifest.json", output / "annotations.csv")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=2, type=Path, default=[
        ROOT / "data/landmarks/novas_5_palavras",
        ROOT / "data/landmarks/novas_5_restantes",
    ])
    parser.add_argument("--output", type=Path, default=ROOT / "data/landmarks/novas_10")
    args = parser.parse_args()
    samples = build_manifest([p.resolve() for p in args.sources], args.output)
    print(json.dumps({"videos": len(samples), "classes": len({s['label'] for s in samples}),
                      "signers": len({s['signer'] for s in samples})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
