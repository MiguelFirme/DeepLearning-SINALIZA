import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  width?: number;
  height?: number;
  onFrame?: (jpeg: Blob) => void;
}

/** Captura JPEGs da webcam. O MediaPipe roda no backend, como no treinamento. */
export function useCameraFrames({ width = 640, height = 480, onFrame }: Options) {
  const videoRef = useRef<HTMLVideoElement>(null!);
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const streamRef = useRef<MediaStream | null>(null);
  const videoFrameIdRef = useRef<number>();
  const animationFrameIdRef = useRef<number>();
  const encodingRef = useRef(false);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [isRunning, setIsRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState("");

  const stop = useCallback(() => {
    if (videoFrameIdRef.current !== undefined && videoRef.current) {
      videoRef.current.cancelVideoFrameCallback(videoFrameIdRef.current);
    }
    if (animationFrameIdRef.current !== undefined) {
      cancelAnimationFrame(animationFrameIdRef.current);
    }
    videoFrameIdRef.current = undefined;
    animationFrameIdRef.current = undefined;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsRunning(false);
    setFps(0);
    encodingRef.current = false;
    const overlayContext = canvasRef.current?.getContext("2d");
    if (overlayContext && canvasRef.current) overlayContext.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  const start = useCallback(async (deviceId?: string) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: 30 },
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
    let lastMediaTime = -1;
    const onNewFrame = (now: number) => {
      if (streamRef.current !== stream) return;
      frames += 1;
      const elapsed = now - fpsStarted;
      if (elapsed >= 1000) {
        setFps(Math.round(frames * 1000 / elapsed));
        frames = 0;
        fpsStarted = now;
      }
      if (!ctx || video.readyState < 2 || encodingRef.current) return;
      ctx.drawImage(video, 0, 0, width, height);
      encodingRef.current = true;
      capture.toBlob((blob) => {
        if (streamRef.current !== stream) return;
        encodingRef.current = false;
        if (blob) onFrameRef.current?.(blob);
      }, "image/jpeg", 0.72);
    };
    if (typeof video.requestVideoFrameCallback === "function") {
      const onVideoFrame: VideoFrameRequestCallback = (now) => {
        if (streamRef.current !== stream) return;
        videoFrameIdRef.current = video.requestVideoFrameCallback(onVideoFrame);
        onNewFrame(now);
      };
      videoFrameIdRef.current = video.requestVideoFrameCallback(onVideoFrame);
    } else {
      // Navegadores antigos: observar o tempo do video para nao duplicar quadros.
      const onAnimationFrame = (now: number) => {
        if (streamRef.current !== stream) return;
        animationFrameIdRef.current = requestAnimationFrame(onAnimationFrame);
        if (video.currentTime === lastMediaTime) return;
        lastMediaTime = video.currentTime;
        onNewFrame(now);
      };
      animationFrameIdRef.current = requestAnimationFrame(onAnimationFrame);
    }
    setIsRunning(true);
  }, [width, height, stop]);

  const switchCamera = useCallback(async (deviceId: string) => {
    stop();
    await start(deviceId);
  }, [stop, start]);

  useEffect(() => () => stop(), [stop]);

  return { videoRef, canvasRef, isRunning, fps, start, stop,
    devices, switchCamera, currentDeviceId };
}
