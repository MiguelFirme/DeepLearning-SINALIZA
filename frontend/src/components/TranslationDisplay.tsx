import { Volume2 } from "lucide-react";
import type { PredictResponse } from "../services/api";
import styles from "./TranslationDisplay.module.css";

interface Props {
  prediction: PredictResponse | null;
  latency: number;
  connected: boolean;
}

export default function TranslationDisplay({ prediction, latency, connected }: Props) {
  const sign = prediction?.sign ?? "—";
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
          {connected ? "Model Ready" : "Disconnected"}
        </span>
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

      {/* Confidence + Latência */}
      <div className={`card ${styles.card}`}>
        <h3 className={styles.cardTitle}>
          <span className={styles.cardIcon}>⚙️</span>
          Camera Controls
        </h3>

        <div className={styles.meter}>
          <span className={styles.meterLabel}>Confidence</span>
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
            <span className={styles.statLabel}>Latency</span>
            <span className={styles.statValue}>{latency}ms</span>
          </div>
          <div>
            <span className={styles.statLabel}>Processing</span>
            <span className={styles.statValue}>
              {prediction?.processing_time_ms?.toFixed(0) ?? 0}ms
            </span>
          </div>
        </div>
      </div>

      {/* Top-K */}
      {prediction && prediction.top_k.length > 1 && (
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
