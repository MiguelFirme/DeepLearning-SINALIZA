import { useCallback, useEffect, useRef, useState } from "react";
import WebcamView, { type WebcamViewHandle } from "../components/WebcamView";
import { useWebSocket } from "../hooks/useWebSocket";
import { api } from "../services/api";
import styles from "./CollectionPage.module.css";

export default function CollectionPage() {
  const webcam = useRef<WebcamViewHandle>(null!);
  const [cameraOn, setCameraOn] = useState(false);
  const [person, setPerson] = useState("");
  const [session, setSession] = useState("");
  const [word, setWord] = useState("");
  const [words, setWords] = useState<string[]>([]);
  const [duration, setDuration] = useState(4);
  const [consent, setConsent] = useState(false);
  const [phase, setPhase] = useState("pronto");
  const [countdown, setCountdown] = useState(3);
  const [remaining, setRemaining] = useState(4);
  const [message, setMessage] = useState("");
  const overlay = useCallback((...args: Parameters<WebcamViewHandle["drawOverlay"]>) => webcam.current?.drawOverlay(...args), []);
  const ws = useWebSocket(overlay, useCallback((data: {status: string; message?: string; path?: string; frames?: number}) => {
    if (data.status === "gravando") {
      setPhase("gravando");
      setMessage("Execute o sinal agora.");
    } else {
      setPhase("pronto");
      setMessage(data.status === "salvo" ? `Salvo: ${data.path} (${data.frames} quadros)`
        : data.message ?? "Gravação cancelada.");
    }
  }, []));

  useEffect(() => {
    api.modelInfo().then((info) => {
      const labels = [...new Set([...info.labels, "_nenhum_sinal"])];
      setWords(labels);
      setWord(labels[0] ?? "_nenhum_sinal");
    }).catch((err) => setMessage(String(err)));
  }, []);

  useEffect(() => {
    if (phase !== "contagem") return;
    if (countdown > 0) {
      const timer = window.setTimeout(() => setCountdown((value) => value - 1), 1000);
      return () => clearTimeout(timer);
    }
    setPhase("aguardando");
    if (!ws.sendControl({action: "record_start", person, session, word, duration, consent})) {
      setPhase("pronto");
      setMessage("Conecte a câmera antes de gravar.");
    }
  }, [phase, countdown, person, session, word, duration, consent, ws.sendControl]);

  useEffect(() => {
    if (phase !== "gravando") return;
    const start = Date.now();
    setRemaining(duration);
    const timer = window.setInterval(() => setRemaining(Math.max(0, Math.ceil(duration - (Date.now() - start) / 1000))), 100);
    // Solicita finalização se não chegar um quadro após o fim da duração.
    const timeout = window.setTimeout(() => ws.sendControl({action: "record_stop"}), (duration + 3) * 1000);
    return () => { clearInterval(timer); clearTimeout(timeout); };
  }, [phase, duration, ws.sendControl]);

  useEffect(() => {
    if (!ws.connected && (phase === "gravando" || phase === "aguardando")) {
      setPhase("pronto");
      setMessage("Conexão interrompida. A gravação incompleta foi descartada; tente novamente.");
    }
  }, [ws.connected, phase]);

  const toggleCamera = async () => {
    if (cameraOn) {
      ws.sendControl({action: "record_cancel"});
      setPhase("pronto");
      webcam.current.stop();
      ws.disconnect();
      setCameraOn(false);
    } else {
      try { await webcam.current.start(); ws.connect(); setCameraOn(true); }
      catch (err) { setMessage(String(err)); }
    }
  };

  const begin = () => {
    if (![person, session].every((value) => /^[A-Za-z0-9_-]{1,64}$/.test(value))) {
      setMessage("Use IDs como P01 e sessao01, com letras, números, _ ou -.");
      return;
    }
    setCountdown(3);
    setPhase("contagem");
    setMessage("Prepare-se: aguarde a contagem e depois execute o sinal.");
  };
  const busy = phase !== "pronto";

  return <div className={`container ${styles.page}`}>
    <h1>Coleta de sinais</h1>
    <p>Grave exemplos com a mesma câmera usada no reconhecimento. Inclua o gesto completo, com rosto, tronco e mãos visíveis.</p>
    <div className={styles.grid}>
      <div>
        <WebcamView ref={webcam} onFrame={ws.sendFrame} canSendFrame={ws.canSendFrame} backendFps={ws.backendFps} />
        <button className="btn btn-outline" onClick={toggleCamera}>{cameraOn ? "Parar câmera" : "Iniciar câmera"}</button>
        <p aria-live="polite" className={styles.countdown}>{phase === "contagem" ? countdown : phase === "gravando" ? `Gravando: ${remaining}s` : ""}</p>
      </div>
      <form className={styles.form} onSubmit={(event) => { event.preventDefault(); begin(); }}>
        <label>Pessoa (ID estável)<input value={person} onChange={(e) => setPerson(e.target.value)} disabled={busy} required maxLength={64} placeholder="P01" /></label>
        <label>Sessão<input value={session} onChange={(e) => setSession(e.target.value)} disabled={busy} required maxLength={64} placeholder="sessao01" /></label>
        <label>Palavra<select value={word} onChange={(e) => setWord(e.target.value)} disabled={busy}>{words.map((label) => <option key={label} value={label}>{label === "_nenhum_sinal" ? "Nenhum sinal (repouso ou movimento fora do vocabulário)" : label}</option>)}</select></label>
        <label>Duração (segundos)<input type="number" min={1} max={30} value={duration} disabled={busy} onChange={(e) => setDuration(Number(e.target.value))} required /></label>
        <label className={styles.consent}><input type="checkbox" checked={consent} disabled={busy} onChange={(e) => setConsent(e.target.checked)} />Concordo com a gravação dos meus pontos corporais para o projeto e confirmo a autorização da pessoa filmada.</label>
        <p>São salvos pontos de corpo, rosto e mãos no computador do backend. A imagem e o áudio não são gravados por esta função.</p>
        <button className="btn btn-primary" disabled={!ws.connected || !consent || busy || !word}>Gravar exemplo</button>
        {busy && <button type="button" className="btn btn-outline" onClick={() => { ws.sendControl({action: "record_cancel"}); setPhase("pronto"); }}>Cancelar e descartar</button>}
        <p role="status">{ws.error ?? message}</p>
      </form>
    </div>
  </div>;
}
