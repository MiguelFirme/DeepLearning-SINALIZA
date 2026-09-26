import { useRef, useState, useCallback, useEffect } from "react";
import type { PredictResponse } from "../services/api";

interface UseWebSocketOptions {
  /** URL do WebSocket (default: ws://localhost:8000/ws/predict). */
  url?: string;
  /** Reconectar automaticamente? */
  autoReconnect?: boolean;
  /** Intervalo de reconexão em ms. */
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  /** Último resultado de predição recebido. */
  prediction: PredictResponse | null;
  /** WebSocket conectado? */
  connected: boolean;
  /** Enviar landmarks para inferência. */
  sendLandmarks: (landmarks: number[][]) => void;
  /** Conectar manualmente. */
  connect: () => void;
  /** Desconectar manualmente. */
  disconnect: () => void;
  /** Latência do último round-trip em ms. */
  latency: number;
}

export function useWebSocket(opts: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    url = `ws://${window.location.hostname}:8000/ws/predict`,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = opts;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const lastSentAt = useRef(0);

  const [connected, setConnected] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [latency, setLatency] = useState(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      console.log("[WS] Conectado");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as PredictResponse;
        setPrediction(data);
        if (lastSentAt.current > 0) {
          setLatency(Date.now() - lastSentAt.current);
        }
      } catch (e) {
        console.warn("[WS] Mensagem inválida:", e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log("[WS] Desconectado");
      if (autoReconnect) {
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] Erro:", err);
      ws.close();
    };

    wsRef.current = ws;
  }, [url, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const sendLandmarks = useCallback((landmarks: number[][]) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    lastSentAt.current = Date.now();
    wsRef.current.send(
      JSON.stringify({ type: "landmarks", data: { landmarks } })
    );
  }, []);

  useEffect(() => {
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return { prediction, connected, sendLandmarks, connect, disconnect, latency };
}

export default useWebSocket;
