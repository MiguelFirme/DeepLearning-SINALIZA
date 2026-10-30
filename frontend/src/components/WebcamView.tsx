import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useCameraFrames } from "../hooks/useCameraFrames";
import type { TrackingOverlay } from "../services/api";
import { drawTrackingOverlay } from "../utils/drawTrackingOverlay";
import { predictTrackingOverlay } from "../utils/predictTrackingOverlay";
import { perfLog, PERF_DETAIL, PERF_LOGGING, recordPerfPaint } from "../utils/perfLogging";
import styles from "./WebcamView.module.css";

interface WebcamViewProps {
  onFrame?: (jpeg: Blob, capturedAt: number, captureEncodeMs: number) => void;
  canSendFrame?: () => boolean;
  width?: number;
  height?: number;
  backendFps?: number;
}

export interface WebcamViewHandle {
  start: () => Promise<void>;
  stop: () => void;
  isRunning: boolean;
  fps: number;
  devices: MediaDeviceInfo[];
  switchCamera: (id: string) => Promise<void>;
  currentDeviceId: string;
  drawOverlay: (overlay: TrackingOverlay | null, ageMs?: number, frameId?: number) => void;
}

const WebcamView = forwardRef<WebcamViewHandle, WebcamViewProps>(
  ({ onFrame, canSendFrame, width = 640, height = 480, backendFps = 0 }, ref) => {
    const mp = useCameraFrames({ width, height, onFrame, canSendFrame });
    const [trackingDetected, setTrackingDetected] = useState(false);
    const lastOverlay = useRef<TrackingOverlay | null>(null);
    const priorOverlay = useRef<TrackingOverlay | null>(null);
    const lastAt = useRef(0);
    const priorAt = useRef(0);
    const lastFrameAge = useRef(0);
    const lastFrameId = useRef(0);
    const paintPending = useRef(false);

    const clearOverlay = () => {
      lastOverlay.current = null;
      priorOverlay.current = null;
      lastAt.current = 0;
      priorAt.current = 0;
      paintPending.current = false;
      setTrackingDetected(false);
      if (mp.canvasRef.current) drawTrackingOverlay(mp.canvasRef.current, null);
    };

    useEffect(() => {
      if (!mp.isRunning) return;
      let animationFrame = 0;
      const paint = (now: number) => {
        const canvas = mp.canvasRef.current;
        const current = lastOverlay.current;
        if (canvas && current && now - lastAt.current <= 250) {
          const paintStarted = PERF_DETAIL && paintPending.current ? performance.now() : 0;
          drawTrackingOverlay(canvas, predictTrackingOverlay(
            priorOverlay.current, current, priorAt.current, lastAt.current,
            now, lastFrameAge.current,
          ));
          if (paintPending.current) {
            paintPending.current = false;
            const visibleMs = lastFrameAge.current + performance.now() - lastAt.current;
            recordPerfPaint(visibleMs);
            if (PERF_DETAIL) perfLog("overlay_visivel", {
              quadro: lastFrameId.current,
              desenho_ms: Number((performance.now() - paintStarted).toFixed(1)),
              captura_ate_tela_ms: Number(visibleMs.toFixed(1)),
            });
          }
        }
        animationFrame = requestAnimationFrame(paint);
      };
      animationFrame = requestAnimationFrame(paint);
      return () => cancelAnimationFrame(animationFrame);
    }, [mp.isRunning, mp.canvasRef]);

    useImperativeHandle(ref, () => ({
      start: () => mp.start(),
      stop: () => { clearOverlay(); mp.stop(); },
      isRunning: mp.isRunning,
      fps: mp.fps,
      devices: mp.devices,
      switchCamera: async (id) => { clearOverlay(); await mp.switchCamera(id); },
      currentDeviceId: mp.currentDeviceId,
      drawOverlay: (overlay, ageMs = 0, frameId = 0) => {
        if (!mp.isRunning || !overlay || !overlay.mask.some(Boolean)) {
          clearOverlay();
          return;
        }
        priorOverlay.current = lastOverlay.current;
        priorAt.current = lastAt.current;
        lastOverlay.current = overlay;
        lastAt.current = performance.now();
        lastFrameAge.current = ageMs;
        lastFrameId.current = frameId;
        paintPending.current = PERF_LOGGING;
        setTrackingDetected(true);
      },
    }));

    return (
      <div className={styles.wrapper}>
        <video
          ref={mp.videoRef}
          className={styles.video}
          width={width}
          height={height}
          playsInline
          muted
        />
        <canvas
          ref={mp.canvasRef}
          className={styles.canvas}
          width={width}
          height={height}
        />

        {/* HUD e legenda dos landmarks rastreados */}
        {mp.isRunning && (
          <>
            <div className={styles.hud}>
              <span>FPS câmera: {mp.fps}</span>
              <span>FPS análise: {backendFps}</span>
              <span>{trackingDetected ? "Pontos detectados" : "Aguardando rastreamento"}</span>
            </div>
            <div className={styles.legend} aria-label="Legenda dos pontos rastreados">
              <span><i className={styles.handLeft} /> Mão esquerda</span>
              <span><i className={styles.handRight} /> Mão direita</span>
              <span><i className={styles.pose} /> Corpo</span>
              <span><i className={styles.face} /> Face</span>
            </div>
          </>
        )}
      </div>
    );
  }
);

WebcamView.displayName = "WebcamView";
export default WebcamView;
