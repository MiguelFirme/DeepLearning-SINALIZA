"""Política editável de seleção do vocabulário do V-LIBRASIL."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.strip().casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


# Termos com aparência de infinitivo que são substantivos, adjetivos ou locuções.
NON_VERBS = {
    "acucar", "borrifador", "cobertor", "computador", "computador portatil",
    "condutor", "cor", "dever de casa", "dolar", "dor de cabeca",
    "dor de garganta", "dor de ouvido", "elementar", "elevador", "flor",
    "interruptor", "jantar (banquete)", "lugar", "maior", "mar (oceano)",
    "melhor", "melhor amigo", "menor", "motor", "mulher", "pior",
    "polegar", "por do sol", "por favor", "por que", "qualquer", "suor",
    "superior", "vapor", "ziper",
}

# Formas e locuções verbais cujo primeiro termo não termina em -ar/-er/-ir.
EXTRA_VERBS = {
    "ato de remar", "caçando", "cale-se", "conseguindo", "devemos",
    "enviando mensagem de texto", "espere", "eu amo voce", "eu vejo",
    "exercite-se", "faz", "ir", "ir embora (partir)", "ir pra casa",
    "lembre-se", "livra-se", "nao poder", "quer", "se afaste",
    "se casar", "se foi (ja era)", "se gabar", "use lingua de sinais",
    "vai", "vamos", "venha",
}

ESSENTIAL_BY_CATEGORY = {
    "escola": {
        "aluno do segundo ano do ensino medio", "atividade", "biblioteca",
        "borracha", "caderno", "calculadora", "campainha", "caneta",
        "carteira", "classe", "colega", "colegio", "computador",
        "dever de casa", "dicionario", "diretoria", "escola", "estudante",
        "estudo", "faculdade", "giz", "lapis", "livro", "matematica",
        "mochila", "papel", "professor", "professora", "quadro", "sala",
        "secretaria", "secretária", "secretário", "teste",
    },
    "casa": {
        "avo", "banheiro", "cama", "casa", "chave", "chuveiro",
        "cozinha", "crianca", "dormitorio", "escada", "familia", "filha",
        "filho", "garagem", "irma", "irmao", "janela", "mae", "pai",
        "porta", "quarto de dormir", "sala", "telefone",
    },
    "necessidades_saude_emergencia": {
        "acidente", "agua", "ajuda", "alimento", "bebida", "com medo",
        "com sede", "comida", "cuidado", "doente", "dor", "dor de cabeca",
        "dor de garganta", "dor de ouvido", "emergencia", "enfermeira",
        "faminto", "fome", "hospital", "medicamento", "medico (doutor)",
        "nariz escorrendo", "perigo", "policial", "remedio", "sangue",
        "sede", "socorro", "tosse", "urgente",
    },
    "interacao": {
        "agora", "amanha", "amigo", "aviso", "boa noite", "bom dia",
        "com licenca", "como", "desculpa", "eu", "hoje", "nao", "nos",
        "obrigado", "oi", "ola", "onde", "por favor", "qual", "quando",
        "quantos", "que", "quem", "sim", "todos", "voce",
    },
    "recepcao": {
        "assinatura", "documento", "endereco", "ficha de registro",
        "formulario", "instituicao", "nome", "numero", "secretaria",
        "secretária", "telefone",
    },
}

_ESSENTIAL = {normalize_label(word) for group in ESSENTIAL_BY_CATEGORY.values() for word in group}
_VERB_FIRST_WORD = re.compile(r"^(?:[a-zà-ÿ]+(?:ar|er|ir|ôr|or))(?=$|[\s(\-])", re.IGNORECASE)


def classify_label(label: str, extra_keep: set[str] | None = None) -> str | None:
    """Retorna motivo da inclusão. A revisão humana dos verbos é recomendada."""
    key = normalize_label(label)
    if key in {normalize_label(x) for x in (extra_keep or set())}:
        return "adicional"
    if key in _ESSENTIAL:
        return "essencial"
    if key in EXTRA_VERBS or (key not in NON_VERBS and _VERB_FIRST_WORD.match(key)):
        return "verbo"
    return None


def read_annotations(path: str | Path) -> dict[str, Counter]:
    """Lê CSV e devolve contagem de vídeos por classe e articulador."""
    result: dict[str, Counter] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "class", "user_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"annotations.csv precisa das colunas {sorted(required)}")
        for row in reader:
            label, signer, video = (row[key].strip() for key in ("class", "user_id", "video_id"))
            if not all((label, signer, video)):
                raise ValueError("annotations.csv contém linha sem classe, articulador ou vídeo")
            result.setdefault(label, Counter())[signer] += 1
    if not result:
        raise ValueError("annotations.csv está vazio")
    return result


def select_vocabulary(labels: set[str], extra_keep: set[str] | None = None) -> dict[str, str]:
    return {label: reason for label in sorted(labels)
            if (reason := classify_label(label, extra_keep)) is not None}


def select_essential_categories(labels: set[str], categories: list[str]) -> dict[str, str]:
    """Seleciona os rótulos do CSV que pertencem às categorias indicadas."""
    if not categories:
        raise ValueError("Informe pelo menos uma categoria")
    unknown = set(categories) - set(ESSENTIAL_BY_CATEGORY)
    if unknown:
        raise ValueError(f"Categorias desconhecidas: {sorted(unknown)}")
    selected: dict[str, str] = {}
    for label in sorted(labels):
        key = normalize_label(label)
        matches = [category for category in dict.fromkeys(categories)
                   if key in {normalize_label(word) for word in ESSENTIAL_BY_CATEGORY[category]}]
        if matches:
            selected[label] = "categoria:" + ",".join(matches)
    return selected


def read_curated_vocabulary(path: str | Path) -> dict[str, str]:
    """Lê grupos de rótulos exatos de um JSON e marca a origem de cada classe."""
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    if not isinstance(groups, dict) or not groups:
        raise ValueError("--class-list precisa ser um objeto JSON de grupos não vazios")
    selected: dict[str, str] = {}
    for category, labels in groups.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("--class-list contém categoria inválida")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Categoria {category!r} precisa de uma lista não vazia")
        for label in labels:
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError(f"Categoria {category!r} contém rótulo inválido")
            if label in selected:
                raise ValueError(f"Classe duplicada em --class-list: {label!r}")
            selected[label] = f"curado:{category}"
    return selected
