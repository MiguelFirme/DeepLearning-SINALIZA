import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  FastForward,
  Maximize,
  Minimize,
  Pause,
  Play,
  Repeat,
  Rewind,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import styles from "./VideoLightbox.module.css";

const SEEK_STEP = 2;
const SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 2];

interface Props {
  src: string;
  poster?: string;
  title: string;
  onClose: () => void;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function VideoLightbox({ src, poster, title, onClose }: Props) {
  const playerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(true);
  const [muted, setMuted] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }, []);

  const seekBy = useCallback((delta: number) => {
    const video = videoRef.current;
    if (!video) return;
    const max = Number.isFinite(video.duration) ? video.duration : Infinity;
    video.currentTime = Math.min(Math.max(video.currentTime + delta, 0), max);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      playerRef.current?.requestFullscreen?.().catch(() => {});
    }
  }, []);

  /* Atalhos de teclado + trava a rolagem da página enquanto aberto */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape":
          if (!document.fullscreenElement) onClose();
          break;
        case " ":
        case "k":
          if (document.activeElement?.tagName === "BUTTON") return;
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowLeft":
          if (document.activeElement instanceof HTMLInputElement) return;
          e.preventDefault();
          seekBy(-SEEK_STEP);
          break;
        case "ArrowRight":
          if (document.activeElement instanceof HTMLInputElement) return;
          e.preventDefault();
          seekBy(SEEK_STEP);
          break;
        case "f":
          toggleFullscreen();
          break;
      }
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose, togglePlay, seekBy, toggleFullscreen]);

  useEffect(() => {
    const onChange = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const changeRate = (value: number) => {
    setRate(value);
    if (videoRef.current) videoRef.current.playbackRate = value;
  };

  const onSeek = (value: number) => {
    setCurrent(value);
    if (videoRef.current) videoRef.current.currentTime = value;
  };

  return createPortal(
    <div
      className={styles.overlay}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Sinal de ${title} em Libras`}
    >
      <div
        ref={playerRef}
        className={styles.player}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <span className={styles.title}>{title}</span>
          <button
            ref={closeRef}
            className={styles.iconBtn}
            onClick={onClose}
            aria-label="Fechar"
            title="Fechar (Esc)"
          >
            <X size={22} />
          </button>
        </div>

        <video
          ref={videoRef}
          className={styles.video}
          src={src}
          poster={poster}
          autoPlay
          playsInline
          loop={loop}
          muted={muted}
          onClick={togglePlay}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        />

        <div className={styles.controls}>
          <input
            type="range"
            className={styles.seek}
            min={0}
            max={duration || 0}
            step={0.01}
            value={Math.min(current, duration || 0)}
            onChange={(e) => onSeek(Number(e.target.value))}
            aria-label="Posição do vídeo"
          />

          <div className={styles.row}>
            <div className={styles.group}>
              <button
                className={styles.iconBtn}
                onClick={() => seekBy(-SEEK_STEP)}
                aria-label={`Voltar ${SEEK_STEP} segundos`}
                title={`Voltar ${SEEK_STEP}s (←)`}
              >
                <Rewind size={20} />
              </button>
              <button
                className={`${styles.iconBtn} ${styles.playBtn}`}
                onClick={togglePlay}
                aria-label={playing ? "Pausar" : "Reproduzir"}
                title={playing ? "Pausar (Espaço)" : "Reproduzir (Espaço)"}
              >
                {playing ? <Pause size={22} /> : <Play size={22} />}
              </button>
              <button
                className={styles.iconBtn}
                onClick={() => seekBy(SEEK_STEP)}
                aria-label={`Avançar ${SEEK_STEP} segundos`}
                title={`Avançar ${SEEK_STEP}s (→)`}
              >
                <FastForward size={20} />
              </button>
              <span className={styles.time}>
                {formatTime(current)} / {formatTime(duration)}
              </span>
            </div>

            <div className={styles.group}>
              <select
                className={styles.speed}
                value={rate}
                onChange={(e) => changeRate(Number(e.target.value))}
                aria-label="Velocidade"
                title="Velocidade"
              >
                {SPEEDS.map((s) => (
                  <option key={s} value={s}>
                    {s}x
                  </option>
                ))}
              </select>
              <button
                className={`${styles.iconBtn} ${loop ? styles.active : ""}`}
                onClick={() => setLoop((v) => !v)}
                aria-label="Repetir"
                aria-pressed={loop}
                title={loop ? "Repetição ligada" : "Repetição desligada"}
              >
                <Repeat size={20} />
              </button>
              <button
                className={styles.iconBtn}
                onClick={() => setMuted((v) => !v)}
                aria-label={muted ? "Ativar som" : "Silenciar"}
                title={muted ? "Ativar som" : "Silenciar"}
              >
                {muted ? <VolumeX size={20} /> : <Volume2 size={20} />}
              </button>
              <button
                className={styles.iconBtn}
                onClick={toggleFullscreen}
                aria-label={fullscreen ? "Sair da tela cheia" : "Tela cheia"}
                title={fullscreen ? "Sair da tela cheia (F)" : "Tela cheia (F)"}
              >
                {fullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
