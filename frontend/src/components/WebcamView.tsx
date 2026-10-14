import { forwardRef, useImperativeHandle, useState } from "react";
import { useCameraFrames } from "../hooks/useCameraFrames";
import type { TrackingOverlay } from "../services/api";
import { drawTrackingOverlay } from "../utils/drawTrackingOverlay";
import styles from "./WebcamView.module.css";

interface WebcamViewProps {
  onFrame?: (jpeg: Blob) => void;
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
  drawOverlay: (overlay: TrackingOverlay | null) => void;
}

const WebcamView = forwardRef<WebcamViewHandle, WebcamViewProps>(
  ({ onFrame, width = 640, height = 480, backendFps = 0 }, ref) => {
    const mp = useCameraFrames({ width, height, onFrame });
    const [trackingDetected, setTrackingDetected] = useState(false);

    useImperativeHandle(ref, () => ({
      start: () => mp.start(),
      stop: () => { mp.stop(); setTrackingDetected(false); },
      isRunning: mp.isRunning,
      fps: mp.fps,
      devices: mp.devices,
      switchCamera: mp.switchCamera,
      currentDeviceId: mp.currentDeviceId,
      drawOverlay: (overlay) => {
        if (mp.canvasRef.current) drawTrackingOverlay(mp.canvasRef.current, mp.isRunning ? overlay : null);
        setTrackingDetected(Boolean(mp.isRunning && overlay?.mask.some(Boolean)));
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
