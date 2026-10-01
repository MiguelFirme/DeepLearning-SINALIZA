"""Variações espaciais pequenas para landmarks, com proveniência explícita."""
from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from ml.data.augmentation import COORD_REGIONS
from ml.data.vocabulary import read_annotations, select_essential_categories


MAX_XY_DELTA = 0.015
SIGNERS = ("Articulador1", "Articulador2", "Articulador3")


def subtle_spatial_variant(
    landmarks: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Preserva sequência, profundidade, visibilidade e partes ausentes.

    A mesma transformação de câmera é aplicada a todos os frames/pontos.
    Um viés minúsculo e fixo por ponto simula erro estável de detecção.
    """
    if landmarks.ndim != 2 or landmarks.shape[1] != 346:
        raise ValueError(f"Landmarks precisam ter shape (T, 346): {landmarks.shape}")
    if mask.shape != (landmarks.shape[0], 4):
        raise ValueError(f"Máscara inválida: {mask.shape}")
    if not np.isfinite(landmarks).all():
        raise ValueError("Landmarks contêm valores não finitos")

    for _ in range(32):
        dx, dy = rng.uniform(-0.004, 0.004, size=2)
        scale = float(rng.uniform(0.997, 1.003))
        angle_deg = float(rng.uniform(-0.75, 0.75))
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        result = landmarks.copy().astype(np.float32)
        largest_delta = 0.0
        for start, end, stride, part in COORD_REGIONS:
            ix = np.arange(start, end, stride)
            iy = ix + 1
            rows = mask[:, part].astype(bool)
            if not rows.any():
                continue
            x = landmarks[np.ix_(rows, ix)]
            y = landmarks[np.ix_(rows, iy)]
            bias = rng.uniform(-0.0005, 0.0005, size=(len(ix), 2))
            xx = 0.5 + scale * ((x - 0.5) * cos_a - (y - 0.5) * sin_a) + dx + bias[:, 0]
            yy = 0.5 + scale * ((x - 0.5) * sin_a + (y - 0.5) * cos_a) + dy + bias[:, 1]
            largest_delta = max(largest_delta, float(np.max(np.abs(xx - x))),
                                float(np.max(np.abs(yy - y))))
            result[np.ix_(rows, ix)] = xx
            result[np.ix_(rows, iy)] = yy
        if 0.0001 < largest_delta <= MAX_XY_DELTA:
            return result, {
                "method": "subtle_spatial_v1",
                "dx": float(dx), "dy": float(dy),
                "scale": scale, "angle_deg": angle_deg,
                "point_bias_bound": 0.0005,
                "max_abs_xy_delta": largest_delta,
            }
    raise ValueError("Não foi possível gerar variação dentro do limite espacial")


def generate_category_variants(
    source_dir: str | Path,
    output_dir: str | Path,
    annotations_path: str | Path,
    categories: list[str],
    variants_per_class: int = 7,
    seed: int = 42,
) -> dict:
    """Cria saída separada: originais intactos + variações rastreáveis."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == source_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("Saída e origem precisam ser diretórios separados")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Saída já contém arquivos: {output_dir}")
    if variants_per_class != 7:
        raise ValueError("Este experimento exige exatamente sete variações por classe")

    with open(source_dir / "manifest.json", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if any(sample.get("is_synthetic", False) for sample in source_manifest):
        raise ValueError("A origem deve conter apenas landmarks reais")
    annotation_counts = read_annotations(annotations_path)
    selected = select_essential_categories(set(annotation_counts), categories)
    labels = sorted(selected)
    if not labels:
        raise ValueError("Nenhuma classe das categorias foi encontrada no CSV")

    samples_by_label: dict[str, dict[str, dict]] = {}
    for sample in source_manifest:
        if sample["label"] not in selected:
            continue
        by_signer = samples_by_label.setdefault(sample["label"], {})
        signer = sample.get("signer")
        if signer in by_signer:
            raise ValueError(f"Mais de um original para {sample['label']} / {signer}")
        by_signer[signer] = sample
    for label in labels:
        if set(samples_by_label.get(label, {})) != set(SIGNERS):
            raise ValueError(f"Classe sem os três articuladores: {label}")
        if any(annotation_counts[label][signer] != 1 for signer in SIGNERS):
            raise ValueError(f"CSV não contém exatamente um original por articulador: {label}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    class_counts: dict[str, dict] = {}
    max_delta = 0.0
    for class_index, label in enumerate(labels):
        counts = {"original": 0, "synthetic": 0}
        for signer_index, signer in enumerate(SIGNERS):
            source = samples_by_label[label][signer]
            relative = Path(source["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Caminho inseguro no manifesto: {relative}")
            source_file = source_dir / relative
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            manifest.append({"path": relative.as_posix(), "label": label, "signer": signer})
            counts["original"] += 1

            with np.load(source_file, allow_pickle=True) as data:
                landmarks = data["landmarks"].astype(np.float32)
                mask = data["mask"].astype(bool)
                metadata = data["metadata"].item()
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if not isinstance(metadata, dict):
                    raise ValueError(f"Metadados inválidos em {source_file}")
            n_variants = 2 + int(signer_index == class_index % len(SIGNERS))
            for number in range(1, n_variants + 1):
                variant, parameters = subtle_spatial_variant(landmarks, mask, rng)
                synthetic_relative = relative.with_name(f"{relative.stem}__syn{number:02d}.npz")
                synthetic_file = output_dir / synthetic_relative
                synthetic_metadata = {**metadata, "augmentation": {
                    **parameters, "source_path": relative.as_posix(), "seed": seed,
                }}
                np.savez_compressed(
                    synthetic_file, landmarks=variant, mask=mask,
                    metadata=np.array([json.dumps(synthetic_metadata, ensure_ascii=False)]),
                )
                manifest.append({
                    "path": synthetic_relative.as_posix(),
                    "label": label, "signer": signer,
                    "is_synthetic": True,
                    "source_path": relative.as_posix(),
                })
                counts["synthetic"] += 1
                max_delta = max(max_delta, parameters["max_abs_xy_delta"])
        if counts != {"original": 3, "synthetic": 7}:
            raise AssertionError(f"Contagem inesperada em {label}: {counts}")
        class_counts[label] = counts

    report = {
        "source_dir": str(source_dir), "output_dir": str(output_dir),
        "annotations": str(Path(annotations_path).resolve()),
        "categories": categories, "seed": seed,
        "variants_per_class": variants_per_class,
        "num_classes": len(labels), "num_original": len(labels) * 3,
        "num_synthetic": len(labels) * 7, "num_total": len(manifest),
        "max_abs_xy_delta": max_delta, "limit_abs_xy_delta": MAX_XY_DELTA,
        "class_counts": class_counts,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "augmentation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
