import type { DictionaryEntry } from "../services/api";
import styles from "./SignCard.module.css";

interface Props {
  entry: DictionaryEntry;
}

export default function SignCard({ entry }: Props) {
  return (
    <div className={styles.card}>
      {/* Placeholder de imagem do sinal */}
      <div className={styles.thumb}>
        <span className={styles.emoji}>🤟</span>
      </div>

      <div className={styles.body}>
        <h4 className={styles.name}>{entry.name}</h4>
        {entry.category && (
          <span className={styles.category}>#{entry.category}</span>
        )}
      </div>
    </div>
  );
}
