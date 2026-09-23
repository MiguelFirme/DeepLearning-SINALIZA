import { useRef, useState, useCallback, useEffect } from "react";

/* ------------------------------------------------------------------ */
/*  Tipos de landmarks                                                */
/* ------------------------------------------------------------------ */

interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface HolisticResults {
  poseLandmarks?: Landmark[];
  leftHandLandmarks?: Landmark[];
  rightHandLandmarks?: Landmark[];
  faceLandmarks?: Landmark[];
}

/* ------------------------------------------------------------------ */
/*  Hook — useMediaPipe                                               */
/* ------------------------------------------------------------------ */

interface UseMediaPipeOptions {
  /** Largura do vídeo. */
  width?: number;
  /** Altura do vídeo. */
  height?: number;
  /** Complexidade do modelo MediaPipe (0, 1, 2). */
  modelComplexity?: 0 | 1 | 2;
  /** FPS alvo para envio de frames. */
  targetFps?: number;
  /** Callback para cada frame de landmarks extraído. */
  onLandmarks?: (flat: number[]) => void;
}

interface UseMediaPipeReturn {
  /** Ref para o <video> HTML. */
  videoRef: React.RefObject<HTMLVideoElement>;
  /** Ref para o <canvas> de overlay. */
  canvasRef: React.RefObject<HTMLCanvasElement>;
  /** Câmera ativa? */
  isRunning: boolean;
  /** FPS atual. */
  fps: number;
  /** Iniciar câmera e detecção. */
  start: () => Promise<void>;
  /** Parar câmera e detecção. */
  stop: () => void;
  /** Lista de dispositivos de vídeo disponíveis. */
  devices: MediaDeviceInfo[];
  /** Trocar câmera por deviceId. */
  switchCamera: (deviceId: string) => Promise<void>;
  /** DeviceId atual. */
  currentDeviceId: string;
}

/**
 * Número total de features por frame:
 *  - left_hand:  21 × 3 =  63
 *  - right_hand: 21 × 3 =  63
 *  - pose:       25 × 4 = 100   (x, y, z, visibility — 25 primeiros landmarks)
 *  - face:       40 × 3 = 120   (contorno de lábios / sobrancelhas selecionados)
 * Total = 346
 */
const POSE_COUNT = 25;
const HAND_COUNT = 21;
const FACE_INDICES = [
  // Lábios externos
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
  317, 14, 87, 178, 88,
  // Sobrancelhas
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  // Contorno do rosto (extra)
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
];

function flattenResults(r: HolisticResults): number[] {
  const out: number[] = [];

  // Left hand (21 × 3)
  for (let i = 0; i < HAND_COUNT; i++) {
    const lm = r.leftHandLandmarks?.[i];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  // Right hand (21 × 3)
  for (let i = 0; i < HAND_COUNT; i++) {
    const lm = r.rightHandLandmarks?.[i];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  // Pose (25 × 4)
  for (let i = 0; i < POSE_COUNT; i++) {
    const lm = r.poseLandmarks?.[i];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0, lm?.visibility ?? 0);
  }

  // Face selecionado (40 × 3)
  for (const idx of FACE_INDICES) {
    const lm = r.faceLandmarks?.[idx];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  return out; // length = 346
}

/* ------------------------------------------------------------------ */

export function useMediaPipe(opts: UseMediaPipeOptions = {}): UseMediaPipeReturn {
  const {
    width = 640,
    height = 480,
    modelComplexity = 1,
    targetFps = 15,
    onLandmarks,
  } = opts;

  const videoRef = useRef<HTMLVideoElement>(null!);
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const streamRef = useRef<MediaStream | null>(null);
  const animIdRef = useRef(0);
  const holisticRef = useRef<any>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState("");

  /* Enumerar câmeras */
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then((devs) => {
      setDevices(devs.filter((d) => d.kind === "videoinput"));
    });
  }, []);

  /* Iniciar MediaPipe Holistic */
  const initHolistic = useCallback(async () => {
    /* @ts-expect-error — import CDN global */
    const { Holistic } = await import("@mediapipe/holistic");

    const holistic = new Holistic({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    return holistic;
  }, [modelComplexity]);

  /* Loop de captura */
  const runLoop = useCallback(
    (holistic: any) => {
      const interval = 1000 / targetFps;
      let lastTime = 0;
      let frameCount = 0;
      let fpsTimer = performance.now();

      const tick = async (now: number) => {
        animIdRef.current = requestAnimationFrame(tick);

        if (now - lastTime < interval) return;
        lastTime = now;

        const video = videoRef.current;
        if (!video || video.readyState < 2) return;

        await holistic.send({ image: video });

        frameCount++;
        if (now - fpsTimer >= 1000) {
          setFps(frameCount);
          frameCount = 0;
          fpsTimer = now;
        }
      };

      holistic.onResults((results: HolisticResults) => {
        /* Desenhar overlay (mãos) */
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx && canvasRef.current) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          drawConnectors(ctx, results);
        }

        /* Flatten e enviar */
        const flat = flattenResults(results);
        onLandmarks?.(flat);
      });

      animIdRef.current = requestAnimationFrame(tick);
    },
    [targetFps, onLandmarks]
  );

  /* Desenhar indicadores visuais básicos */
  function drawConnectors(ctx: CanvasRenderingContext2D, results: HolisticResults) {
    ctx.lineWidth = 2;

    const drawHand = (landmarks: Landmark[] | undefined, color: string) => {
      if (!landmarks) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      for (const lm of landmarks) {
        const x = lm.x * ctx.canvas.width;
        const y = lm.y * ctx.canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    };

    drawHand(results.leftHandLandmarks, "#00e676");
    drawHand(results.rightHandLandmarks, "#ff4081");
  }

  /* ---------- API pública ---------- */

  const start = useCallback(
    async (deviceId?: string) => {
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: width },
          height: { ideal: height },
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }

      const track = stream.getVideoTracks()[0];
      setCurrentDeviceId(track.getSettings().deviceId ?? "");

      const holistic = await initHolistic();
      holisticRef.current = holistic;
      runLoop(holistic);
      setIsRunning(true);
    },
    [width, height, initHolistic, runLoop]
  );

  const stop = useCallback(() => {
    cancelAnimationFrame(animIdRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    holisticRef.current?.close();
    holisticRef.current = null;
    setIsRunning(false);
    setFps(0);
  }, []);

  const switchCamera = useCallback(
    async (deviceId: string) => {
      stop();
      await start(deviceId);
    },
    [stop, start]
  );

  /* Cleanup */
  useEffect(() => {
    return () => stop();
  }, [stop]);

  return {
    videoRef,
    canvasRef,
    isRunning,
    fps,
    start,
    stop,
    devices,
    switchCamera,
    currentDeviceId,
  };
}

export default useMediaPipe;
