"""Montagem de frases em português a partir de sinais isolados de Libras.

Usa a API do Google Gemini via HTTP (httpx, já está no requirements.txt). A chave
fica somente no backend. Se não houver chave ou a chamada falhar, devolve um
fallback simples (palavras unidas) para a interface nunca ficar travada.
"""
from __future__ import annotations

import logging

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """Você ajuda a traduzir sinais de Libras para português do Brasil.
Um modelo de visão reconheceu SINAIS ISOLADOS, na ordem em que foram feitos. Libras tem
gramática própria: o sinalizador omite artigos, preposições e conjugações, e o modelo
pode ter errado alguma palavra.

Sua tarefa: transformar a sequência de palavras em UMA frase natural e curta em português.

Regras:
- Use somente o sentido das palavras recebidas. Não invente fatos, nomes ou detalhes novos.
- Pode acrescentar palavras de ligação (eu, ir, ao, de, a, o...) e conjugar verbos.
- Se uma palavra não combinar com as outras, ela pode ser um erro de reconhecimento;
  ignore-a só se for claramente incoerente.
- "Vontade" costuma indicar desejo ou necessidade ("estou com vontade de...", "quero...").
- Responda APENAS com a frase final, sem aspas, explicações ou marcadores."""


def _fallback(words: list[str]) -> str:
    text = " ".join(w.strip().lower() for w in words if w.strip())
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _extract_text(data: dict) -> str:
    """Pega o texto da primeira resposta do Gemini (ou '' se bloqueada/vazia)."""
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


async def build_sentence(words: list[str]) -> tuple[str, str]:
    """Retorna (frase, origem). origem: 'llm' ou 'fallback'."""
    clean = [w.strip()[:40] for w in words if w and w.strip()][-settings.llm_max_words:]
    if not clean:
        return "", "fallback"
    if not settings.gemini_api_key:
        logger.warning("SINALIZA_GEMINI_API_KEY não definida; usando fallback sem IA.")
        return _fallback(clean), "fallback"

    def make_payload(with_thinking: bool) -> dict:
        generation_config: dict = {
            "temperature": 0.3,
            # Folga para o modelo: em alguns modelos o "raciocínio" interno conta aqui.
            "maxOutputTokens": 1024,
        }
        if with_thinking:
            # Raciocínio mínimo (resposta rápida). Modelos antigos podem não aceitar.
            generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Sinais reconhecidos: " + ", ".join(clean)}]}
            ],
            "generationConfig": generation_config,
        }

    url = API_URL.format(model=settings.llm_model)
    headers = {"x-goog-api-key": settings.gemini_api_key, "content-type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
            res = await client.post(url, json=make_payload(True), headers=headers)
            if res.status_code == 400:
                # Alguns modelos não aceitam thinkingLevel: tenta de novo sem ele.
                logger.warning("Gemini recusou thinkingConfig (%s); tentando sem.", res.text[:200])
                res = await client.post(url, json=make_payload(False), headers=headers)
        if res.status_code != 200:
            # Mostra o motivo real (chave inválida, cota, modelo inexistente...)
            logger.error("Gemini respondeu %s: %s", res.status_code, res.text[:300])
        else:
            text = _extract_text(res.json())
            if text:
                return text.strip('"“” '), "llm"
            logger.warning("Gemini não devolveu texto (resposta bloqueada ou vazia); usando fallback.")
    except Exception as exc:  # rede, timeout...
        logger.error("Falha ao chamar o Gemini: %s", exc)
    return _fallback(clean), "fallback"