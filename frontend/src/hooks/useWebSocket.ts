import { useRef, useState, useCallback, useEffect } from "react";
import type { PredictResponse, TrackingOverlay } from "../services/api";
import { flushPerfSummary, perfLog, recordPerfFrame, resetPerfSummary, PERF_DETAIL, PERF_LOGGING } from "../utils/perfLogging";

interface PendingFrame {
  id: number;
  capturedAt: number;
  sentAt: number;
  captureEncodeMs: number;
  bytes: number;
}

interface WsMessage {
  type: "ready" | "prediction" | "error" | "control" | "pong" | "frame_ack";
  data: any;
}

export function useWebSocket(onOverlay?: (overlay: TrackingOverlay | null, ageMs?: number, frameId?: number) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const onOverlayRef = useRef(onOverlay);
  onOverlayRef.current = onOverlay;
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const shouldReconnect = useRef(false);
  const sentAt = useRef<PendingFrame[]>([]);
  const nextFrameId = useRef(0);
  const serverReady = useRef(false);
  const framesInFlight = useRef(0);
  const processedFrames = useRef(0);
  const frameProcessingTotal = useRef(0);
  const roundTripTotal = useRef(0);
  const [connected, setConnected] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [latency, setLatency] = useState(0);
  const [backendFps, setBackendFps] = useState(0);
  const [frameProcessingMs, setFrameProcessingMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    shouldReconnect.current = true;
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;
    clearTimeout(reconnectTimer.current);
    serverReady.current = false;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/predict`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return;
      try {
        const message = JSON.parse(event.data) as WsMessage;
        if (message.type === "ready") {
          resetPerfSummary();
          serverReady.current = true;
          framesInFlight.current = 0;
          sentAt.current = [];
          nextFrameId.current = 0;
          processedFrames.current = 0;
          frameProcessingTotal.current = 0;
          roundTripTotal.current = 0;
          onOverlayRef.current?.(null);
          setConnected(true);
          setError(message.data?.overlay === true ? null : "Backend sem dados dos pontos. Atualize o repositório e reinicie o backend.");
        } else if (message.type === "frame_ack") {
          framesInFlight.current = Math.max(0, framesInFlight.current - 1);
          const frame = sentAt.current.shift();
          const acknowledgedAt = performance.now();
          const ageMs = frame === undefined ? undefined : acknowledgedAt - frame.capturedAt;
          if (ageMs !== undefined) roundTripTotal.current += ageMs;
          processedFrames.current += 1;
          frameProcessingTotal.current += Number(message.data?.frame_processing_ms) || 0;
          onOverlayRef.current?.(message.data?.overlay ?? null, ageMs, frame?.id);
          if (frame) {
            const backendMs = Number(message.data?.frame_processing_ms) || 0;
            const transitMs = acknowledgedAt - frame.sentAt;
            const stages = message.data?.profile;
            recordPerfFrame({
              jpegMs: frame.captureEncodeMs,
              communicationMs: Math.max(0, transitMs - backendMs),
              mediapipeMs: stages ? Number(stages.mediapipe_ms) : null,
              modelMs: stages?.prediction_ran ? Number(stages.inference_ms) : null,
            });
            if (PERF_DETAIL) perfLog("quadro", {
              quadro: frame.id,
              quadro_backend: Number(message.data?.frame_id) || 0,
              jpeg_kb: Number((frame.bytes / 1024).toFixed(1)),
              captura_jpeg_ms: Number(frame.captureEncodeMs.toFixed(1)),
              captura_ate_envio_ms: Number((frame.sentAt - frame.capturedAt).toFixed(1)),
              envio_ate_ack_ms: Number(transitMs.toFixed(1)),
              backend_total_ms: Number(backendMs.toFixed(1)),
              decode_ms: stages ? Number(stages.decode_ms.toFixed(1)) : null,
              mediapipe_ms: stages ? Number(stages.mediapipe_ms.toFixed(1)) : null,
              preparo_ms: stages ? Number(stages.prepare_ms.toFixed(1)) : null,
              inferencia_ms: stages ? Number(stages.inference_ms.toFixed(1)) : null,
              janela_quadros: stages ? Number(stages.window_frames) : null,
              classificou: stages ? Boolean(stages.prediction_ran) : null,
              comunicacao_e_outros_ms: Number(Math.max(0, transitMs - backendMs).toFixed(1)),
              captura_ate_ack_ms: Number((acknowledgedAt - frame.capturedAt).toFixed(1)),
            });
          }
        } else if (message.type === "prediction") {
          setPrediction(message.data as PredictResponse);
        } else if (message.type === "error") {
          setError(message.data?.message ?? "Erro no backend");
        }
      } catch {
        setError("Resposta inválida do backend");
      }
    };
    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      wsRef.current = null;
      resetPerfSummary();
      serverReady.current = false;
      setConnected(false);
      onOverlayRef.current?.(null);
      framesInFlight.current = 0;
      sentAt.current = [];
      processedFrames.current = 0;
      frameProcessingTotal.current = 0;
      roundTripTotal.current = 0;
      setLatency(0);
      setBackendFps(0);
      setFrameProcessingMs(0);
      if (shouldReconnect.current) {
        reconnectTimer.current = setTimeout(connect, 3000);
      }
    };
    ws.onerror = () => {
      setError("Não foi possível conectar à API na porta 8000");
      ws.close();
    };
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnect.current = false;
    clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
    resetPerfSummary();
    serverReady.current = false;
    setConnected(false);
    setPrediction(null);
    onOverlayRef.current?.(null);
    framesInFlight.current = 0;
    sentAt.current = [];
    processedFrames.current = 0;
    frameProcessingTotal.current = 0;
    roundTripTotal.current = 0;
    setLatency(0);
    setBackendFps(0);
    setFrameProcessingMs(0);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const count = processedFrames.current;
      setBackendFps(count);
      if (count) {
        setFrameProcessingMs(Math.round(frameProcessingTotal.current / count));
        setLatency(Math.round(roundTripTotal.current / count));
      }
      processedFrames.current = 0;
      frameProcessingTotal.current = 0;
      roundTripTotal.current = 0;
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!PERF_LOGGING) return;
    const timer = window.setInterval(flushPerfSummary, 3000);
    return () => window.clearInterval(timer);
  }, []);

  const canSendFrame = useCallback(() => {
    const ws = wsRef.current;
    return Boolean(serverReady.current && ws && ws.readyState === WebSocket.OPEN &&
      framesInFlight.current === 0 && ws.bufferedAmount < 1_000_000);
  }, []);

  const sendFrame = useCallback((jpeg: Blob, capturedAt: number, captureEncodeMs: number) => {
    const ws = wsRef.current;
    if (!ws || !canSendFrame()) return;
    const id = ++nextFrameId.current;
    sentAt.current.push({ id, capturedAt, sentAt: performance.now(), captureEncodeMs, bytes: jpeg.size });
    framesInFlight.current += 1;
    ws.send(jpeg);
  }, [canSendFrame]);

  useEffect(() => () => {
    shouldReconnect.current = false;
    clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
  }, []);

  return { prediction, connected, error, sendFrame, canSendFrame, connect, disconnect, latency, backendFps, frameProcessingMs };
}
