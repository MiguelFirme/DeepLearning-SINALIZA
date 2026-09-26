"""
Divisão de dados com prevenção de vazamento.

Estratégias:
    - stratified: StratifiedKFold, mantém proporção de classes.
    - group: GroupShuffleSplit por signer (quando disponível).
    - Verificação de vazamento: garante zero sobreposição.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    strategy: str = "stratified"  # stratified | group
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    random_seed: int = 42
    group_key: str = "signer"
    min_samples_per_class: int = 1


class DataSplitter:
    """Divide amostras em train/val/test com verificação de vazamento."""

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()

    def split(self, samples: list[dict]) -> dict[str, list[int]]:
        """
        Divide lista de amostras e retorna dicionário com índices:
        {"train": [...], "val": [...], "test": [...]}.
        """
        cfg = self.config
        labels = [s["label"] for s in samples]
        groups = [s.get(cfg.group_key, "unknown") for s in samples]

        has_groups = any(g != "unknown" for g in groups)
        use_group = cfg.strategy == "group" and has_groups

        if use_group:
            result = self._group_split(labels, groups)
        else:
            result = self._stratified_split(labels)

        self._verify_no_leakage(result)
        self._log_stats(result, labels)
        return result

    def split_by_signer(
        self,
        samples: list[dict],
        test_signer: str,
        train_signers: list[str] | tuple[str, ...] | None = None,
        allowed_labels: set[str] | None = None,
    ) -> dict[str, list[int]]:
        """Separa articuladores de treino e teste sem criar validação vazada.

        Os índices retornados sempre se referem ao manifesto original. Amostras
        de articuladores não selecionados ou de classes fora de
        ``allowed_labels`` são ignoradas.
        """
        available_signers = sorted({
            str(sample.get(self.config.group_key, "unknown"))
            for sample in samples
            if sample.get(self.config.group_key, "unknown") != "unknown"
        })
        if test_signer not in available_signers:
            raise ValueError(
                f"Articulador de teste desconhecido: {test_signer}. "
                f"Disponíveis: {available_signers}"
            )

        if train_signers is None:
            selected_train_signers = [s for s in available_signers if s != test_signer]
        else:
            selected_train_signers = list(dict.fromkeys(train_signers))

        if not selected_train_signers:
            raise ValueError("Informe ao menos um articulador de treino.")
        if test_signer in selected_train_signers:
            raise ValueError("O articulador de teste não pode aparecer no treino.")

        unknown_train = sorted(set(selected_train_signers) - set(available_signers))
        if unknown_train:
            raise ValueError(
                f"Articuladores de treino desconhecidos: {unknown_train}. "
                f"Disponíveis: {available_signers}"
            )

        labels_filter = set(allowed_labels) if allowed_labels is not None else None
        train_signers_set = set(selected_train_signers)
        train_idx: list[int] = []
        test_idx: list[int] = []
        for index, sample in enumerate(samples):
            label = sample["label"]
            if labels_filter is not None and label not in labels_filter:
                continue
            signer = sample.get(self.config.group_key, "unknown")
            if signer in train_signers_set:
                train_idx.append(index)
            elif signer == test_signer:
                test_idx.append(index)

        result = {"train": train_idx, "val": [], "test": test_idx}
        if not train_idx or not test_idx:
            raise ValueError(
                "Split por articulador vazio: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

        self._verify_no_leakage(result)
        train_groups = {
            samples[index].get(self.config.group_key, "unknown") for index in train_idx
        }
        test_groups = {
            samples[index].get(self.config.group_key, "unknown") for index in test_idx
        }
        group_overlap = train_groups & test_groups
        if group_overlap:
            raise ValueError(f"Vazamento de articuladores: {sorted(group_overlap)}")

        logger.info(
            "Signer holdout: train=%s, val=[], test=%s",
            sorted(train_groups),
            sorted(test_groups),
        )
        self._log_stats(result, [s["label"] for s in samples])
        return result

    def labels_with_signer_coverage(
        self,
        samples: list[dict],
        required_signers: list[str] | tuple[str, ...],
    ) -> list[str]:
        """Retorna classes que possuem amostra de todos os articuladores pedidos."""
        required = set(required_signers)
        if not required:
            raise ValueError("A cobertura requer ao menos um articulador.")

        signers_by_label: dict[str, set[str]] = {}
        for sample in samples:
            signers_by_label.setdefault(sample["label"], set()).add(
                str(sample.get(self.config.group_key, "unknown"))
            )
        return sorted(
            label for label, signers in signers_by_label.items()
            if required.issubset(signers)
        )

    def _stratified_split(self, labels: list[str]) -> dict[str, list[int]]:
        """Split estratificado — mantém proporção de classes em cada partição."""
        rng = np.random.default_rng(self.config.random_seed)
        label_indices: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            label_indices.setdefault(lab, []).append(i)

        train_idx, val_idx, test_idx = [], [], []
        for lab, indices in sorted(label_indices.items()):
            idx = np.array(indices)
            rng.shuffle(idx)
            n = len(idx)
            n_test = max(self.config.min_samples_per_class, int(n * self.config.test_ratio))
            n_val = max(self.config.min_samples_per_class, int(n * self.config.val_ratio))
            n_test = min(n_test, n - 1) if n > 1 else 0
            n_val = min(n_val, n - n_test - 1) if n - n_test > 1 else 0

            test_idx.extend(idx[:n_test].tolist())
            val_idx.extend(idx[n_test : n_test + n_val].tolist())
            train_idx.extend(idx[n_test + n_val :].tolist())

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def _group_split(self, labels: list[str], groups: list[str]) -> dict[str, list[int]]:
        """Split por grupo (signer) — impede que o mesmo signer apareça em splits diferentes."""
        rng = np.random.default_rng(self.config.random_seed)
        unique_groups = sorted(set(groups))
        rng.shuffle(unique_groups)
        n = len(unique_groups)
        n_test = max(1, int(n * self.config.test_ratio))
        n_val = max(1, int(n * self.config.val_ratio))

        test_groups = set(unique_groups[:n_test])
        val_groups = set(unique_groups[n_test : n_test + n_val])
        train_groups = set(unique_groups[n_test + n_val :])

        train_idx, val_idx, test_idx = [], [], []
        for i, g in enumerate(groups):
            if g in test_groups:
                test_idx.append(i)
            elif g in val_groups:
                val_idx.append(i)
            else:
                train_idx.append(i)

        logger.info(
            "Group split: train=%s, val=%s, test=%s",
            sorted(train_groups),
            sorted(val_groups),
            sorted(test_groups),
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    @staticmethod
    def _verify_no_leakage(result: dict[str, list[int]]):
        """Verifica que não há sobreposição de índices entre splits."""
        sets = {k: set(v) for k, v in result.items()}
        for a, sa in sets.items():
            for b, sb in sets.items():
                if a >= b:
                    continue
                overlap = sa & sb
                if overlap:
                    raise ValueError(f"VAZAMENTO detectado entre {a} e {b}: {len(overlap)} amostras em comum!")
        logger.info("✓ Nenhum vazamento detectado entre splits.")

    @staticmethod
    def _log_stats(result: dict[str, list[int]], labels: list[str]):
        total = sum(len(v) for v in result.values())
        for split_name, indices in result.items():
            n = len(indices)
            pct = n / total * 100 if total else 0
            split_labels = [labels[i] for i in indices]
            n_classes = len(set(split_labels))
            logger.info(f"  {split_name}: {n} amostras ({pct:.1f}%), {n_classes} classes")

    def save_split(self, result: dict[str, list[int]], path: str | Path):
        """Salva split em JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Split salvo em {path}")

    @staticmethod
    def load_split(path: str | Path) -> dict[str, list[int]]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
