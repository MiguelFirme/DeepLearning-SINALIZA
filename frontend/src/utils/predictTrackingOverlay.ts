import type { TrackingOverlay } from "../services/api";

/** Apenas para desenhar: os valores previstos nunca são enviados ao modelo. */
export function predictTrackingOverlay(
  previous: TrackingOverlay | null,
  current: TrackingOverlay,
  previousAt: number,
  currentAt: number,
  now: number,
  frameAgeMs: number,
): TrackingOverlay {
  if (!previous || previous.landmarks.length !== 346 || current.landmarks.length !== 346 ||
      previous.mask.length !== 4 || current.mask.length !== 4) return current;

  const interval = currentAt - previousAt;
  if (interval <= 0 || interval > 350 || now - currentAt > 250 || frameAgeMs > 350) return current;
  // Antecipação visual máxima de 80 ms, limitada a 80% do último deslocamento.
  const lead = Math.min(80, Math.max(0, frameAgeMs + now - currentAt));
  const factor = Math.min(0.8, lead / Math.max(40, interval));
  const landmarks = current.landmarks.slice();

  const advance = (offset: number, count: number, stride: number) => {
    for (let index = 0; index < count; index += 1) {
      for (let axis = 0; axis < 2; axis += 1) {
        const position = offset + index * stride + axis;
        const last = current.landmarks[position];
        const prior = previous.landmarks[position];
        if (!Number.isFinite(last) || !Number.isFinite(prior)) continue;
        // Um salto de detecção não deve projetar os pontos longe da pessoa.
        const motion = Math.max(-0.04, Math.min(0.04, (last - prior) * factor));
        landmarks[position] = Math.max(0, Math.min(1, last + motion));
      }
    }
  };
  if (previous.mask[0] && current.mask[0]) advance(0, 21, 3);
  if (previous.mask[1] && current.mask[1]) advance(63, 21, 3);
  if (previous.mask[2] && current.mask[2]) advance(126, 25, 4);
  if (previous.mask[3] && current.mask[3]) advance(226, 40, 3);
  return { landmarks, mask: current.mask };
}
