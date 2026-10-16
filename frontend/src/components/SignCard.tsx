import { useEffect, useRef, useState } from "react";
import type { DictionaryEntry } from "../services/api";
import styles from "./SignCard.module.css";

/**
 * Como o vídeo do sinal toca:
 *  - "visible": em loop enquanto o card está na tela (parece um GIF).
 *  - "hover":   só ao passar o mouse; no celular, toca/pausa ao tocar no card.
 * Em ambos os casos o vídeo só é baixado quando precisa tocar (preload="none").
 * Quem pediu "reduzir movimento" no sistema sempre usa o modo "hover".
 */
const PLAY_MODE: "visible" | "hover" = "hover";

interface Props {
  entry: DictionaryEntry;
}

export default function SignCard({ entry }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);

  const hasVideo = Boolean(entry.example_video_url) && !failed;
  const hasImage = Boolean(entry.image_url);
  const [reduceMotion] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  const [canHover] = useState(() => window.matchMedia("(hover: hover)").matches);
  const hoverMode = PLAY_MODE === "hover" || reduceMotion;

  /* Modo "visible": toca quando entra na tela, pausa quando sai */
  useEffect(() => {
    const video = videoRef.current;
    if (!video || hoverMode) return;
    const observer = new IntersectionObserver(
      ([item]) => {
        if (item.isIntersecting) video.play().catch(() => {});
        else video.pause();
      },
      { threshold: 0.4 }
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, [hasVideo, hoverMode]);

  const play = () => {
    videoRef.current?.play().catch(() => {});
  };
  const stop = () => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = 0;
  };
  const toggle = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) play();
    else stop();
  };

  return (
    <div className={styles.card}>
      <div
        className={styles.thumb}
        onMouseEnter={hoverMode && canHover ? play : undefined}
        onMouseLeave={hoverMode && canHover ? stop : undefined}
        onClick={hoverMode && !canHover && hasVideo ? toggle : undefined}
      >
        {hasVideo ? (
          <video
            ref={videoRef}
            className={styles.video}
            src={entry.example_video_url ?? undefined}
            poster={entry.poster_url ?? undefined}
            muted
            loop
            playsInline
            preload="none"
            aria-label={`Sinal de ${entry.name} em Libras`}
            onError={() => setFailed(true)}
          />
        ) : hasImage ? (
          <img
            className={styles.video}
            src={entry.image_url ?? undefined}
            alt={`Sinal de ${entry.name} em Libras`}
            loading="lazy"
          />
        ) : (
          <span className={styles.emoji}>🤟</span>
        )}
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
