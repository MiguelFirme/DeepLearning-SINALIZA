import { useRef, useCallback, useState, useEffect } from "react";
import { Camera, CameraOff } from "lucide-react";
import WebcamView, { type WebcamViewHandle } from "../components/WebcamView";
import TranslationDisplay from "../components/TranslationDisplay";
import SentenceBuilder from "../components/SentenceBuilder";
import { useWebSocket } from "../hooks/useWebSocket";
import styles from "./HomePage.module.css";

export default function HomePage() {
  const webcamRef = useRef<WebcamViewHandle>(null!);
  const handleOverlay = useCallback((overlay: Parameters<WebcamViewHandle["drawOverlay"]>[0], ageMs?: number, frameId?: number) => {
    webcamRef.current?.drawOverlay(overlay, ageMs, frameId);
  }, []);
  const ws = useWebSocket(handleOverlay);

  const [cameraOn, setCameraOn] = useState(false);
  const [sentence, setSentence] = useState<string[]>([]);
  const handleFrame = useCallback((jpeg: Blob, capturedAt: number, captureEncodeMs: number) =>
    ws.sendFrame(jpeg, capturedAt, captureEncodeMs), [ws.sendFrame]);

  /* Sinal que o usuário acabou de apagar: não volta por alguns segundos,
     mesmo que o modelo continue reconhecendo o mesmo sinal. */
  const ignoredRef = useRef<{ sign: string; until: number } | null>(null);
  const IGNORE_MS = 4000;

  /* Montar frase progressivamente */
  useEffect(() => {
    const sign = ws.prediction?.sign;
    if (!sign || ws.prediction?.is_cooldown) return;
    const ignored = ignoredRef.current;
    if (ignored && ignored.sign === sign && Date.now() < ignored.until) return;
    setSentence((prev) => {
      if (prev[prev.length - 1] === sign) return prev;
      return [...prev.slice(-19), sign];
    });
  }, [ws.prediction]);

  /* Apagar uma palavra específica (ex.: reconhecida errado) */
  const removeWord = useCallback((index: number) => {
    setSentence((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      ignoredRef.current = { sign: prev[index], until: Date.now() + IGNORE_MS };
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  /* Limpar todas as palavras */
  const clearSentence = useCallback(() => {
    const current = ws.prediction?.sign;
    if (current) ignoredRef.current = { sign: current, until: Date.now() + IGNORE_MS };
    setSentence([]);
  }, [ws.prediction]);

  /* Toggle da câmera */
  const toggleCamera = async () => {
    if (cameraOn) {
      webcamRef.current?.stop();
      ws.disconnect();
      setCameraOn(false);
    } else {
      try {
        await webcamRef.current?.start();
        ws.connect();
        setCameraOn(true);
        setSentence([]);
      } catch (err) {
        console.error("Não foi possível iniciar a câmera:", err);
      }
    }
  };

  // A frase contém somente palavras aceitas; os avisos atuais aparecem abaixo.
  const displaySentence = sentence.join(", ");
  const currentStatus = ws.latestPrediction?.status;
  const confirming = ws.latestPrediction?.sign && ws.latestPrediction.sign !== ws.prediction?.sign;

  return (
    <div className={`container ${styles.page}`}>
      <div className={styles.grid}>
        {/* Coluna esquerda — Vídeo */}
        <div className={styles.videoCol}>
          <WebcamView ref={webcamRef} onFrame={handleFrame} canSendFrame={ws.canSendFrame} backendFps={ws.backendFps} />

          {/* Sentence overlay */}
          <div className={styles.sentenceBar}>
            <p className={styles.sentenceText}>
              {displaySentence || "Aguardando tradução..."}
            </p>
            <span className={styles.sentenceHint}>
              {cameraOn
                ? !ws.connected
                  ? "Conectando ao serviço de tradução…"
                  : currentStatus === "poor_tracking"
                    ? "Não foi possível rastrear o corpo — ajuste o enquadramento"
                    : currentStatus === "no_sign"
                      ? "Nenhum sinal reconhecido nesta janela"
                    : currentStatus === "idle"
                      ? "Aguardando movimento das mãos e braços"
                      : currentStatus === "recognized"
                        ? confirming ? "Confirmando sinal…" : "Sinal reconhecido"
                        : currentStatus === "uncertain"
                          ? "Sinal não identificado com confiança suficiente"
                          : "Capturando sinais e analisando movimentos…"
                : "Câmera desligada"}
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
          <SentenceBuilder words={sentence} onRemove={removeWord} onClear={clearSentence} />
          <TranslationDisplay
            prediction={ws.prediction}
            latestPrediction={ws.latestPrediction}
            latency={ws.latency}
            frameProcessingMs={ws.frameProcessingMs}
            connected={ws.connected}
            error={ws.error}
            cameraOn={cameraOn}
          />
        </div>
      </div>
    </div>
  );
}
