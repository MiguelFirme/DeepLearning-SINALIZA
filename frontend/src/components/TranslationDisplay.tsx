import { Volume2 } from "lucide-react";
import type { PredictResponse } from "../services/api";
import styles from "./TranslationDisplay.module.css";

interface Props {
  prediction: PredictResponse | null;
  latency: number;
  frameProcessingMs: number;
  connected: boolean;
  error: string | null;
  cameraOn: boolean;
}

export default function TranslationDisplay({ prediction, latency, frameProcessingMs, connected, error, cameraOn }: Props) {
  const translationStatus = error
    ? { label: "Erro na tradução", detail: error, tone: "error" }
    : !cameraOn
      ? { label: "Câmera desligada", detail: "Inicie a câmera para começar a tradução.", tone: "neutral" }
      : !connected
        ? { label: "Conectando", detail: "Estabelecendo conexão com o serviço de tradução.", tone: "warning" }
        : prediction?.status === "poor_tracking"
          ? { label: "Rastreamento insuficiente", detail: "Mantenha o tronco e o rosto visíveis e ajuste o enquadramento.", tone: "warning" }
          : prediction?.status === "idle"
            ? { label: "Aguardando sinal", detail: "Faça o sinal em frente à câmera.", tone: "neutral" }
            : prediction?.status === "recognized" && prediction.sign
              ? { label: "Tradução identificada", detail: "O modelo reconheceu um sinal.", tone: "success" }
              : prediction?.status === "uncertain"
                ? { label: "Analisando sinal", detail: "O sinal ainda não foi identificado com confiança suficiente.", tone: "warning" }
                : { label: "Capturando sinais", detail: "Câmera ativa. Os quadros estão sendo enviados para análise.", tone: "active" };

  const sign = prediction?.status === "poor_tracking" ? "Ajuste o enquadramento"
    : prediction?.status === "uncertain" ? "Sinal incerto"
    : prediction?.sign ?? "—";
  const confidence = prediction?.confidence ?? 0;
  const pct = Math.round(confidence * 100);

  /* Text-to-speech (pt-BR) */
  const speak = () => {
    if (!prediction?.sign) return;
    const u = new SpeechSynthesisUtterance(prediction.sign);
    u.lang = "pt-BR";
    u.rate = 0.9;
    speechSynthesis.speak(u);
  };

  return (
    <div className={styles.panel}>
      {/* Status badge */}
      <div className={styles.status}>
        <span
          className={`badge ${connected ? "badge-success" : "badge-error"}`}
        >
          <span
            className={styles.dot}
            style={{ background: connected ? "#10b981" : "#ef4444" }}
          />
          {connected ? "Modelo pronto" : "Desconectado"}
        </span>
        {error && <p role="alert">{error}</p>}
      </div>

      {/* Estado explícito da tradução */}
      <div className={`card ${styles.card} ${styles.stateCard}`} data-tone={translationStatus.tone} aria-live="polite">
        <h3 className={styles.cardTitle}>
          <span className={styles.cardIcon}>●</span>
          Estado da tradução
        </h3>
        <p className={styles.stateTitle}>{translationStatus.label}</p>
        <p className={styles.stateDetail}>{translationStatus.detail}</p>
      </div>

      {/* Última Tradução */}
      <div className={`card ${styles.card}`}>
        <h3 className={styles.cardTitle}>
          <span className={styles.cardIcon}>✋</span>
          Última Tradução
        </h3>
        <p className={styles.signText}>{sign}</p>

        {prediction?.sign && (
          <button className={`btn btn-outline ${styles.speakBtn}`} onClick={speak}>
            <Volume2 size={16} />
            Leitura de voz
          </button>
        )}
      </div>

      {/* Confiança + Latência */}
      <div className={`card ${styles.card}`}>
        <h3 className={styles.cardTitle}>
          <span className={styles.cardIcon}>⚙️</span>
          Controles da câmera
        </h3>

        <div className={styles.meter}>
          <span className={styles.meterLabel}>Confiança</span>
          <div className={styles.bar}>
            <div
              className={styles.barFill}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className={styles.meterValue}>{pct}%</span>
        </div>

        <div className={styles.stats}>
          <div>
            <span className={styles.statLabel}>Ida e volta/quadro</span>
            <span className={styles.statValue}>{latency}ms</span>
          </div>
          <div>
            <span className={styles.statLabel}>Transformer</span>
            <span className={styles.statValue}>
              {prediction?.processing_time_ms?.toFixed(0) ?? 0}ms
            </span>
          </div>
          <div>
            <span className={styles.statLabel}>Backend/quadro</span>
            <span className={styles.statValue}>{frameProcessingMs}ms</span>
          </div>
        </div>
      </div>

      {/* Top-K */}
      {prediction?.sign && prediction.top_k.length > 1 && (
        <div className={`card ${styles.card}`}>
          <h3 className={styles.cardTitle}>Top Candidatos</h3>
          <ul className={styles.topList}>
            {prediction.top_k.map((item, i) => (
              <li key={item.class_id} className={styles.topItem}>
                <span className={styles.rank}>#{i + 1}</span>
                <span className={styles.topSign}>{item.sign}</span>
                <span className={styles.topConf}>
                  {Math.round(item.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
