#!/usr/bin/env python3
"""Resume a fusão de 25 classes por origem e expõe erros das dez classes novas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def origin_of(path: str) -> str:
    if path.startswith("../minds_libras_all20/"):
        return "minds_15"
    if path.startswith("../novas_5_palavras/") or path.startswith("../novas_5_restantes/"):
        return "novas_10"
    raise ValueError(f"Origem desconhecida: {path}")


def summarize(folder: Path) -> dict:
    result = {}
    for split in ("val", "test"):
        for first in ("pose_face", "full"):
            path = folder / f"{split}_{first}_predictions.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            summary = {str(k): {} for k in (3, 5)}
            for top_k in (3, 5):
                selected = [row for row in rows if int(row["top_k"]) == top_k]
                if len(selected) != (85):
                    raise ValueError(f"Esperados 85 vídeos em {path} para top {top_k}")
                for origin in ("minds_15", "novas_10"):
                    group = [row for row in selected if origin_of(row["path"]) == origin]
                    expected = 75 if origin == "minds_15" else 10
                    if len(group) != expected:
                        raise ValueError(f"{origin} em {path}: {len(group)} em vez de {expected}")
                    item = {
                        "total": len(group),
                        "pose_face_correct": sum(r["pose_prediction"] == r["true"] for r in group),
                        "full_correct": sum(r["full_prediction"] == r["true"] for r in group),
                        "fusion_correct": sum(r["fused_prediction"] == r["true"] for r in group),
                    }
                    if origin == "novas_10":
                        item["predictions"] = [
                            {"class": r["true"], "pose_face": r["pose_prediction"],
                             "full": r["full_prediction"], "fusion": r["fused_prediction"]}
                            for r in sorted(group, key=lambda r: r["true"])
                        ]
                    summary[str(top_k)][origin] = item
            result[f"{split}_{first}"] = summary
    output = folder / "subgroup_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/fusion_mixed25"))
    args = parser.parse_args()
    result = summarize(args.input)
    for key, values in result.items():
        print(key, {top: {origin: {k: v for k, v in stats.items() if k != "predictions"}
                           for origin, stats in by_origin.items()}
                    for top, by_origin in values.items()})


if __name__ == "__main__":
    main()
