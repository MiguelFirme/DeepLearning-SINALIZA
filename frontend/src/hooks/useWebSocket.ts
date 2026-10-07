import { useRef, useState, useCallback, useEffect } from "react";
import type { PredictResponse } from "../services/api";

interface WsMessage {
  type: "ready" | "prediction" | "error" | "control" | "pong" | "frame_ack";
  data: any;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const shouldReconnect = useRef(false);
  const lastSentAt = useRef(0);
  const framesInFlight = useRef(0);
  const [connected, setConnected] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [latency, setLatency] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    shouldReconnect.current = true;
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;
    clearTimeout(reconnectTimer.current);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/predict`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WsMessage;
        if (message.type === "ready") {
          framesInFlight.current = 0;
          setConnected(true);
          setError(null);
        } else if (message.type === "frame_ack") {
          framesInFlight.current = Math.max(0, framesInFlight.current - 1);
        } else if (message.type === "prediction") {
          setPrediction(message.data as PredictResponse);
          if (lastSentAt.current) setLatency(Date.now() - lastSentAt.current);
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
      setConnected(false);
      framesInFlight.current = 0;
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
    setConnected(false);
    setPrediction(null);
    framesInFlight.current = 0;
  }, []);

  const sendFrame = useCallback((jpeg: Blob) => {
    const ws = wsRef.current;
    if (!connected || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (framesInFlight.current >= 2 || ws.bufferedAmount > 1_000_000) return;
    lastSentAt.current = Date.now();
    framesInFlight.current += 1;
    ws.send(jpeg);
  }, [connected]);

  useEffect(() => () => {
    shouldReconnect.current = false;
    clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
  }, []);

  return { prediction, connected, error, sendFrame, connect, disconnect, latency };
}
