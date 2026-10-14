import type { TrackingOverlay } from "../services/api";

// Mesma ordem de ml/features/landmarks.py: 21 + 21 maos, 25 pose, 40 face.
const POSE_INDICES = [
  0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
];

type Point = [number, number];

export function drawTrackingOverlay(canvas: HTMLCanvasElement, overlay?: TrackingOverlay | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!overlay || overlay.landmarks.length !== 346 || overlay.mask.length !== 4) return;

  const point = (offset: number, stride: number, index: number): Point | null => {
    const x = overlay.landmarks[offset + index * stride];
    const y = overlay.landmarks[offset + index * stride + 1];
    return Number.isFinite(x) && Number.isFinite(y)
      ? [x * canvas.width, y * canvas.height] : null;
  };
  const draw = (points: Array<Point | null>, connections: number[][], color: string, radius: number) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    for (const [a, b] of connections) {
      const from = points[a];
      const to = points[b];
      if (!from || !to) continue;
      ctx.beginPath();
      ctx.moveTo(...from);
      ctx.lineTo(...to);
      ctx.stroke();
    }
    for (const position of points) {
      if (!position) continue;
      ctx.beginPath();
      ctx.arc(position[0], position[1], radius, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  if (overlay.mask[0]) {
    draw(Array.from({ length: 21 }, (_, i) => point(0, 3, i)), HAND_CONNECTIONS, "#00e676", 3);
  }
  if (overlay.mask[1]) {
    draw(Array.from({ length: 21 }, (_, i) => point(63, 3, i)), HAND_CONNECTIONS, "#ff4081", 3);
  }
  if (overlay.mask[2]) {
    const pose = POSE_INDICES.map((_, i) => point(126, 4, i));
    const connections = POSE_CONNECTIONS.map(([a, b]) => [POSE_INDICES.indexOf(a), POSE_INDICES.indexOf(b)]);
    draw(pose, connections, "#38bdf8", 3);
  }
  if (overlay.mask[3]) {
    draw(Array.from({ length: 40 }, (_, i) => point(226, 3, i)), [], "#fbbf24", 2);
  }
}
