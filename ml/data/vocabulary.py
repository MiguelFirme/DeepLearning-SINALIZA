"""Política editável de seleção do vocabulário do V-LIBRASIL."""
from __future__ import annotations

import csv
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
