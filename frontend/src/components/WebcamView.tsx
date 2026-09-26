import { forwardRef, useImperativeHandle } from "react";
import { useMediaPipe } from "../hooks/useMediaPipe";
import styles from "./WebcamView.module.css";

interface WebcamViewProps {
  onLandmarks?: (flat: number[]) => void;
  width?: number;
  height?: number;
}

export interface WebcamViewHandle {
  start: () => Promise<void>;
  stop: () => void;
  isRunning: boolean;
  fps: number;
  devices: MediaDeviceInfo[];
  switchCamera: (id: string) => Promise<void>;
  currentDeviceId: string;
}

const WebcamView = forwardRef<WebcamViewHandle, WebcamViewProps>(
  ({ onLandmarks, width = 640, height = 480 }, ref) => {
    const mp = useMediaPipe({ width, height, onLandmarks, targetFps: 15 });

    useImperativeHandle(ref, () => ({
      start: () => mp.start(),
      stop: mp.stop,
      isRunning: mp.isRunning,
      fps: mp.fps,
      devices: mp.devices,
      switchCamera: mp.switchCamera,
      currentDeviceId: mp.currentDeviceId,
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

        {/* HUD overlay */}
        {mp.isRunning && (
          <div className={styles.hud}>
            <span>FPS: {mp.fps}</span>
          </div>
        )}
      </div>
    );
  }
);

WebcamView.displayName = "WebcamView";
export default WebcamView;
