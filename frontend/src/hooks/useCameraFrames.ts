import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  width?: number;
  height?: number;
  targetFps?: number;
  onFrame?: (jpeg: Blob) => void;
}

/** Captura JPEGs da webcam. O MediaPipe roda no backend, como no treinamento. */
export function useCameraFrames({ width = 640, height = 480, targetFps = 10, onFrame }: Options) {
  const videoRef = useRef<HTMLVideoElement>(null!);
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const encodingRef = useRef(false);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [isRunning, setIsRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState("");

  const stop = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = undefined;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsRunning(false);
    setFps(0);
  }, []);

  const start = useCallback(async (deviceId?: string) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: width },
        height: { ideal: height },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });
    streamRef.current = stream;
    const video = videoRef.current;
    if (!video) {
      stop();
      throw new Error("Elemento de vídeo indisponível");
    }
    video.srcObject = stream;
    try {
      await video.play();
    } catch (error) {
      stop();
      throw error;
    }
    setCurrentDeviceId(stream.getVideoTracks()[0]?.getSettings().deviceId ?? "");
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    setDevices(allDevices.filter((item) => item.kind === "videoinput"));

    const capture = document.createElement("canvas");
    capture.width = width;
    capture.height = height;
    const ctx = capture.getContext("2d");
    let frames = 0;
    let fpsStarted = performance.now();
    timerRef.current = setInterval(() => {
      if (!ctx || video.readyState < 2 || encodingRef.current) return;
      ctx.drawImage(video, 0, 0, width, height);
      encodingRef.current = true;
      capture.toBlob((blob) => {
        encodingRef.current = false;
        if (blob && streamRef.current === stream) onFrameRef.current?.(blob);
      }, "image/jpeg", 0.72);
      frames += 1;
      const now = performance.now();
      if (now - fpsStarted >= 1000) {
        setFps(frames);
        frames = 0;
        fpsStarted = now;
      }
    }, 1000 / targetFps);
    setIsRunning(true);
  }, [width, height, targetFps, stop]);

  const switchCamera = useCallback(async (deviceId: string) => {
    stop();
    await start(deviceId);
  }, [stop, start]);

  useEffect(() => () => stop(), [stop]);

  return { videoRef, canvasRef, isRunning, fps, start, stop,
    devices, switchCamera, currentDeviceId };
}
