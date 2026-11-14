import { useState } from "react";
import { Sparkles, Trash2, Undo2, Volume2, X } from "lucide-react";
import api from "../services/api";
import styles from "./SentenceBuilder.module.css";

interface Props {
  /** Sinais reconhecidos até agora, em ordem. */
  words: string[];
  /** Remove um sinal específico (pelo índice na lista). */
  onRemove: (index: number) => void;
  /** Limpa todos os sinais. */
  onClear: () => void;
}

interface BuiltSentence {
  sentence: string;
  source: "llm" | "fallback";
  /** Palavras usadas para montar esta frase (para saber se ficou desatualizada). */
  basedOn: string;
}

export default function SentenceBuilder({ words, onRemove, onClear }: Props) {
  const [result, setResult] = useState<BuiltSentence | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wordsKey = words.join("\u0001");
  /* A frase só vale enquanto as palavras forem as mesmas usadas para montá-la. */
  const visible = result && words.length > 0 ? result : null;
  const stale = visible !== null && visible.basedOn !== wordsKey;

  const build = async () => {
    if (words.length === 0 || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.sentence(words);
      setResult({ ...res, basedOn: wordsKey });
    } catch (err) {
      console.error(err);
      setError("Não foi possível montar a frase. Verifique se o backend está rodando.");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setResult(null);
    setError(null);
    onClear();
  };

  const speak = () => {
    if (!visible?.sentence) return;
    const u = new SpeechSynthesisUtterance(visible.sentence);
    u.lang = "pt-BR";
    u.rate = 0.9;
    speechSynthesis.speak(u);
  };

  return (
    <div className={`card ${styles.card}`}>
      <h3 className={styles.title}>
        <span>💬</span> Montar frase
      </h3>

      <div className={styles.words} aria-label="Sinais reconhecidos">
        {words.length === 0 ? (
          <span className={styles.empty}>Nenhum sinal reconhecido ainda.</span>
        ) : (
          words.map((w, i) => (
            <span key={`${w}-${i}`} className={styles.chip}>
              {w}
              <button
                type="button"
                className={styles.chipRemove}
                onClick={() => onRemove(i)}
                disabled={loading}
                aria-label={`Remover ${w}`}
                title={`Remover "${w}"`}
              >
                <X size={12} />
              </button>
            </span>
          ))
        )}
      </div>

      <div className={styles.actions}>
        <button className="btn btn-primary" onClick={build} disabled={words.length === 0 || loading}>
          <Sparkles size={16} />
          {loading ? "Montando…" : "Montar frase"}
        </button>
        <button
          className="btn btn-outline"
          onClick={() => onRemove(words.length - 1)}
          disabled={words.length === 0 || loading}
          title="Remove a última palavra"
        >
          <Undo2 size={16} />
          Desfazer
        </button>
        <button className="btn btn-outline" onClick={clearAll} disabled={words.length === 0 || loading}>
          <Trash2 size={16} />
          Limpar
        </button>
      </div>

      {error && <p className={styles.error} role="alert">{error}</p>}

      {visible && (
        <div className={styles.result} aria-live="polite">
          <p className={`${styles.sentence} ${stale ? styles.stale : ""}`}>{visible.sentence}</p>
          {stale && (
            <p className={styles.note}>As palavras mudaram. Clique em "Montar frase" para atualizar.</p>
          )}
          {!stale && visible.source === "fallback" && (
            <p className={styles.note}>
              Frase simples, sem IA. Configure a chave da LLM no backend (arquivo .env) para uma tradução melhor.
            </p>
          )}
          <button className={`btn btn-outline ${styles.speak}`} onClick={speak} disabled={stale}>
            <Volume2 size={16} />
            Leitura de voz
          </button>
        </div>
      )}
    </div>
  );
}