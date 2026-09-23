import { useRef, useCallback, useState, useEffect } from "react";
import { Camera, CameraOff } from "lucide-react";
import WebcamView, { type WebcamViewHandle } from "../components/WebcamView";
import TranslationDisplay from "../components/TranslationDisplay";
import { useWebSocket } from "../hooks/useWebSocket";
import styles from "./HomePage.module.css";

const SEQUENCE_LEN = 48;
const FEATURES_PER_FRAME = 346;

export default function HomePage() {
  const webcamRef = useRef<WebcamViewHandle>(null!);
  const ws = useWebSocket();

  const [cameraOn, setCameraOn] = useState(false);
  const [sentence, setSentence] = useState<string[]>([]);
  const bufferRef = useRef<number[][]>([]);

  /* Acumular frames e enviar quando buffer cheio */
  const handleLandmarks = useCallback(
    (flat: number[]) => {
      if (flat.length !== FEATURES_PER_FRAME) return;
      bufferRef.current.push(flat);

      if (bufferRef.current.length >= SEQUENCE_LEN) {
        ws.sendLandmarks(bufferRef.current);
        bufferRef.current = [];
      }
    },
    [ws]
  );

  /* Montar frase progressivamente */
  useEffect(() => {
    if (ws.prediction?.sign && !ws.prediction.is_cooldown) {
      setSentence((prev) => {
        const last = prev[prev.length - 1];
        if (last === ws.prediction!.sign) return prev;
        return [...prev.slice(-19), ws.prediction!.sign!];
      });
    }
  }, [ws.prediction]);

  /* Toggle da câmera */
  const toggleCamera = async () => {
    if (cameraOn) {
      webcamRef.current?.stop();
      ws.disconnect();
      setCameraOn(false);
      bufferRef.current = [];
    } else {
      await webcamRef.current?.start();
      ws.connect();
      setCameraOn(true);
      setSentence([]);
    }
  };

  const displaySentence = sentence.join(", ");

  return (
    <div className={`container ${styles.page}`}>
      <div className={styles.grid}>
        {/* Coluna esquerda — Vídeo */}
        <div className={styles.videoCol}>
          <WebcamView ref={webcamRef} onLandmarks={handleLandmarks} />

          {/* Sentence overlay */}
          <div className={styles.sentenceBar}>
            <p className={styles.sentenceText}>
              {displaySentence || "Aguardando tradução..."}
            </p>
            <span className={styles.sentenceHint}>
              {cameraOn ? "translating real-time…" : "câmera desligada"}
            </span>
          </div>

          {/* Controle de câmera */}
          <div className={styles.controls}>
            <button
              className={`btn ${cameraOn ? "btn-outline" : "btn-primary"}`}
              onClick={toggleCamera}
            >
              {cameraOn ? <CameraOff size={18} /> : <Camera size={18} />}
              {cameraOn ? "Parar" : "Iniciar Câmera"}
            </button>

            {cameraOn && webcamRef.current && (
              <select
                className={styles.select}
                value={webcamRef.current.currentDeviceId}
                onChange={(e) => webcamRef.current.switchCamera(e.target.value)}
              >
                {webcamRef.current.devices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Camera ${d.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {/* Coluna direita — Painel */}
        <div className={styles.sideCol}>
          <TranslationDisplay
            prediction={ws.prediction}
            latency={ws.latency}
            connected={ws.connected}
          />
        </div>
      </div>
    </div>
  );
}
