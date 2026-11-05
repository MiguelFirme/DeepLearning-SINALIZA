import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Reconhecimento de voz via Web Speech API (nativa do navegador).
 *
 * - Funciona no Chrome, Edge e Safari. Firefox não suporta (`supported = false`).
 * - Exige HTTPS ou localhost, e permissão de microfone.
 * - No Chrome o áudio é processado pelo serviço de voz do Google, então precisa de internet.
 *
 * O TypeScript 5.5 ainda não traz os tipos dessa API, por isso são declarados aqui.
 */

interface AlternativeLike {
  transcript: string;
}

interface ResultLike {
  isFinal: boolean;
  length: number;
  [index: number]: AlternativeLike;
}

interface RecognitionEventLike {
  resultIndex: number;
  results: { length: number; [index: number]: ResultLike };
}

interface RecognitionErrorEventLike {
  error: string;
}

interface RecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: RecognitionErrorEventLike) => void) | null;
  onresult: ((event: RecognitionEventLike) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type RecognitionConstructor = new () => RecognitionLike;

function getRecognitionConstructor(): RecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/* O dicionário guarda os números como 0.png … 9.png; o reconhecedor às vezes devolve por extenso. */
const NUMBER_WORDS: Record<string, string> = {
  zero: "0",
  um: "1",
  uma: "1",
  dois: "2",
  duas: "2",
  tres: "3",
  quatro: "4",
  cinco: "5",
  seis: "6",
  sete: "7",
  oito: "8",
  nove: "9",
};

/** Limpa o texto falado para virar termo de busca: sem pontuação, espaços extras e "dois" → "2". */
export function cleanTranscript(raw: string): string {
  const text = raw
    .replace(/[.,;:!?¿¡"“”()]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const folded = text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
  return NUMBER_WORDS[folded] ?? text;
}

const ERROR_MESSAGES: Record<string, string> = {
  "not-allowed": "Permita o uso do microfone no navegador para buscar por voz.",
  "service-not-allowed": "Permita o uso do microfone no navegador para buscar por voz.",
  "no-speech": "Não ouvi nada. Toque no microfone e tente de novo.",
  "audio-capture": "Nenhum microfone foi encontrado.",
  network:
    "Sem acesso ao serviço de voz. Verifique a internet. O Brave bloqueia esse recurso; use Chrome ou Edge.",
  "language-not-supported": "Este navegador não reconhece português por voz.",
};

/* Tempos de segurança: o botão nunca fica preso, mesmo que o navegador não avise o fim. */
const STOP_GRACE_MS = 1200; // espera o resultado final depois de pedir para parar
const SESSION_MAX_MS = 30000; // limite de uma sessão de escuta

interface Options {
  /** Idioma do reconhecimento (padrão: português do Brasil). */
  lang?: string;
  /** Chamado uma vez com o texto final já limpo, quando a pessoa termina de falar. */
  onResult: (text: string) => void;
}

export function useSpeechRecognition({ lang = "pt-BR", onResult }: Options) {
  const [supported] = useState(() => getRecognitionConstructor() !== null);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<RecognitionLike | null>(null);
  const timersRef = useRef<{ grace?: number; session?: number }>({});
  // guarda o callback mais recente sem precisar recriar o reconhecedor
  const onResultRef = useRef(onResult);
  useEffect(() => {
    onResultRef.current = onResult;
  }, [onResult]);

  const clearTimers = useCallback(() => {
    window.clearTimeout(timersRef.current.grace);
    window.clearTimeout(timersRef.current.session);
    timersRef.current = {};
  }, []);

  /** Encerra tudo de forma definitiva, sem depender de nenhum evento do navegador. */
  const teardown = useCallback(() => {
    clearTimers();
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    if (recognition) {
      recognition.onstart = null;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      try {
        recognition.abort();
      } catch {
        /* já estava parado */
      }
    }
    setListening(false);
    setInterim("");
  }, [clearTimers]);

  /** Pede para parar, mas ainda aceita o resultado final; se o navegador não responder, força. */
  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    setListening(false); // o botão já responde ao clique
    try {
      recognition.stop();
    } catch {
      /* ignora */
    }
    window.clearTimeout(timersRef.current.grace);
    timersRef.current.grace = window.setTimeout(teardown, STOP_GRACE_MS);
  }, [teardown]);

  const start = useCallback(() => {
    const Recognition = getRecognitionConstructor();
    if (!Recognition || recognitionRef.current) return;

    const recognition = new Recognition();
    recognition.lang = lang;
    recognition.continuous = false; // para sozinho quando a pessoa para de falar
    recognition.interimResults = true; // mostra o texto enquanto fala
    recognition.maxAlternatives = 1;

    // ignora eventos de uma sessão antiga que já foi substituída/encerrada
    const isCurrent = () => recognitionRef.current === recognition;

    recognition.onstart = () => {
      if (!isCurrent()) return;
      setError(null);
      setInterim("");
      setListening(true);
    };

    recognition.onresult = (event) => {
      if (!isCurrent()) return;
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0]?.transcript ?? "";
        if (result.isFinal) finalText += transcript;
        else interimText += transcript;
      }
      if (finalText) {
        setInterim("");
        const cleaned = cleanTranscript(finalText);
        if (cleaned) onResultRef.current(cleaned);
        // a sessão já acabou; garante o encerramento mesmo sem o evento "end"
        window.clearTimeout(timersRef.current.grace);
        timersRef.current.grace = window.setTimeout(teardown, STOP_GRACE_MS);
      } else {
        setInterim(interimText);
      }
    };

    recognition.onerror = (event) => {
      if (!isCurrent()) return;
      // "aborted" acontece quando a gente mesmo cancela; não é erro para mostrar
      if (event.error === "aborted") return;
      console.warn("[busca por voz] erro:", event.error);
      setError(ERROR_MESSAGES[event.error] ?? "Não foi possível usar a busca por voz.");
      teardown();
    };

    recognition.onend = () => {
      if (!isCurrent()) return;
      teardown();
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      console.warn("[busca por voz] não foi possível iniciar:", err);
      teardown();
      setError("Não foi possível iniciar o microfone.");
      return;
    }

    // limite de segurança: nenhuma sessão fica ouvindo para sempre
    timersRef.current.session = window.setTimeout(() => {
      if (!isCurrent()) return;
      teardown();
      setError("O navegador não respondeu. Tente de novo ou use Chrome/Edge.");
    }, SESSION_MAX_MS);
  }, [lang, teardown]);

  /** Clique no microfone: ouvindo → para; travado/iniciando → cancela; parado → começa. */
  const toggle = useCallback(() => {
    if (listening) stop();
    else if (recognitionRef.current) teardown();
    else start();
  }, [listening, start, stop, teardown]);

  /* Ao sair da página, solta o microfone */
  useEffect(() => teardown, [teardown]);

  return { supported, listening, interim, error, start, stop, toggle };
}