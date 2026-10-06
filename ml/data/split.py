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
    strategy: str = "group"  # stratified (legado) | group
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    random_seed: int = 42
    group_key: str = "video"  # video | session | signer
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
        if cfg.strategy == "group":
            groups = [self.group_id(s) for s in samples]
            result = self._group_split(labels, groups)
            self.verify_group_isolation(samples, result)
        elif cfg.strategy == "stratified":
            result = self._stratified_split(labels)
        else:
            raise ValueError(f"Estratégia desconhecida: {cfg.strategy}")

        self._verify_no_leakage(result)
        self.verify_group_isolation(samples, result, group_key="video")
        self._log_stats(result, labels)
        return result

    def split_by_signer(
        self,
        samples: list[dict],
        test_signer: str,
        train_signers: list[str] | tuple[str, ...] | None = None,
        allowed_labels: set[str] | None = None,
        val_signer: str | None = None,
    ) -> dict[str, list[int]]:
        """Separa articuladores de treino e teste sem criar validação vazada.

        Os índices retornados sempre se referem ao manifesto original. Amostras
        de articuladores não selecionados ou de classes fora de
        ``allowed_labels`` são ignoradas.
        """
        available_signers = sorted({
            str(sample.get("signer", "unknown"))
            for sample in samples
            if sample.get("signer", "unknown") != "unknown"
        })
        if test_signer not in available_signers:
            raise ValueError(
                f"Articulador de teste desconhecido: {test_signer}. "
                f"Disponíveis: {available_signers}"
            )

        if train_signers is None:
            selected_train_signers = [s for s in available_signers if s not in {test_signer, val_signer}]
        else:
            selected_train_signers = list(dict.fromkeys(train_signers))

        if not selected_train_signers:
            raise ValueError("Informe ao menos um articulador de treino.")
        if test_signer in selected_train_signers:
            raise ValueError("O articulador de teste não pode aparecer no treino.")
        if val_signer is not None and (val_signer == test_signer or val_signer in selected_train_signers):
            raise ValueError("O articulador de validação deve ser distinto dos de treino e teste.")
        if val_signer is not None and val_signer not in available_signers:
            raise ValueError(f"Articulador de validação desconhecido: {val_signer}")

        unknown_train = sorted(set(selected_train_signers) - set(available_signers))
        if unknown_train:
            raise ValueError(
                f"Articuladores de treino desconhecidos: {unknown_train}. "
                f"Disponíveis: {available_signers}"
            )

        labels_filter = set(allowed_labels) if allowed_labels is not None else None
        train_signers_set = set(selected_train_signers)
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for index, sample in enumerate(samples):
            label = sample["label"]
            if labels_filter is not None and label not in labels_filter:
                continue
            signer = sample.get("signer", "unknown")
            if signer in train_signers_set:
                train_idx.append(index)
            elif signer == val_signer and not sample.get("is_synthetic", False):
                val_idx.append(index)
            elif signer == test_signer and not sample.get("is_synthetic", False):
                test_idx.append(index)

        result = {"train": train_idx, "val": val_idx, "test": test_idx}
        if not train_idx or not test_idx:
            raise ValueError(
                "Split por articulador vazio: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

        self._verify_no_leakage(result)
        train_groups = {
            samples[index].get("signer", "unknown") for index in train_idx
        }
        test_groups = {
            samples[index].get("signer", "unknown") for index in test_idx
        }
        val_groups = {samples[index].get("signer", "unknown") for index in val_idx}
        group_overlap = (train_groups & test_groups) | (train_groups & val_groups) | (test_groups & val_groups)
        if group_overlap:
            raise ValueError(f"Vazamento de articuladores: {sorted(group_overlap)}")
        self.verify_group_isolation(samples, result, group_key="video")

        logger.info(
            "Signer holdout: train=%s, val=%s, test=%s",
            sorted(train_groups),
            sorted(val_groups),
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
                str(sample.get("signer", "unknown"))
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
        """Duas divisões GroupShuffleSplit, preservando cada grupo integralmente."""
        if len(set(groups)) < 3 or not 0 < self.config.test_ratio < 1 or not 0 < self.config.val_ratio < 1:
            raise ValueError("Group split requer ao menos três grupos e proporções válidas")
        try:
            from sklearn.model_selection import GroupShuffleSplit
        except ImportError:
            GroupShuffleSplit = None

        indices = np.arange(len(groups))
        def split_once(candidate_indices, fraction, seed):
            candidate_groups = np.asarray(groups)[candidate_indices]
            if GroupShuffleSplit is not None:
                splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                left, right = next(splitter.split(candidate_indices, groups=candidate_groups))
                return candidate_indices[left], candidate_indices[right]
            # Mesmo contrato de divisão por grupo quando sklearn não está instalado.
            unique = np.unique(candidate_groups)
            if len(unique) < 2:
                raise ValueError("Grupos insuficientes para treino/validação/teste")
            np.random.RandomState(seed).shuffle(unique)
            n_right = max(1, min(len(unique) - 1, int(np.ceil(fraction * len(unique)))))
            right_groups = set(unique[:n_right])
            right = np.asarray([g in right_groups for g in candidate_groups])
            return candidate_indices[~right], candidate_indices[right]

        remaining, test = split_once(indices, self.config.test_ratio, self.config.random_seed)
        relative_val = self.config.val_ratio / (1 - self.config.test_ratio)
        train, val = split_once(remaining, relative_val, self.config.random_seed + 1)
        train_idx = train.tolist()
        val_idx = val.tolist()
        test_idx = test.tolist()
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        test_groups = {groups[i] for i in test_idx}

        logger.info(
            "Group split: train=%s, val=%s, test=%s",
            sorted(train_groups),
            sorted(val_groups),
            sorted(test_groups),
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def group_id(self, sample: dict, group_key: str | None = None) -> str:
        key = group_key or self.config.group_key
        if key == "video":
            value = sample.get("source_path") or sample.get("path")
        elif key == "session":
            value = sample.get("session") or sample.get("session_id")
        else:
            value = sample.get(key)
        if value is None or str(value).strip() in ("", "unknown"):
            raise ValueError(f"Grupo '{key}' ausente no manifesto: {sample.get('path')}")
        return str(value)

    def verify_group_isolation(self, samples: list[dict], result: dict[str, list[int]],
                               group_key: str | None = None) -> None:
        owners: dict[str, str] = {}
        for split_name, indices in result.items():
            for index in indices:
                group = self.group_id(samples[index], group_key)
                previous = owners.setdefault(group, split_name)
                if previous != split_name:
                    raise ValueError(f"Vazamento de grupo '{group}' entre {previous} e {split_name}")

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
