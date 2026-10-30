const perfMode = typeof window === "undefined" ? null :
  new URLSearchParams(window.location.search).get("perf");

/** ?perf=1 mostra resumos; ?perf=full também mostra cada quadro. */
export const PERF_LOGGING = perfMode === "1" || perfMode === "full";
export const PERF_DETAIL = perfMode === "full";

interface FrameTimes {
  jpegMs: number;
  communicationMs: number;
  mediapipeMs: number | null;
  modelMs: number | null;
}

interface Totals {
  startedAt: number;
  frames: number;
  jpegMs: number;
  communicationMs: number;
  mediapipeMs: number;
  profiledFrames: number;
  modelMs: number;
  modelCalls: number;
  visibleMs: number;
  visibleFrames: number;
}

let cameraFps = 0;
const emptyTotals = (): Totals => ({
  startedAt: performance.now(), frames: 0, jpegMs: 0,
  communicationMs: 0, mediapipeMs: 0, profiledFrames: 0,
  modelMs: 0, modelCalls: 0, visibleMs: 0, visibleFrames: 0,
});
let totals: Totals | null = null;

export function resetPerfSummary() {
  totals = null;
  cameraFps = 0;
}

export function recordCameraFps(fps: number) {
  if (PERF_LOGGING) cameraFps = fps;
}

export function recordPerfFrame(frame: FrameTimes) {
  if (!PERF_LOGGING) return;
  totals ??= emptyTotals();
  totals.frames += 1;
  totals.jpegMs += frame.jpegMs;
  totals.communicationMs += frame.communicationMs;
  if (frame.mediapipeMs !== null) {
    totals.mediapipeMs += frame.mediapipeMs;
    totals.profiledFrames += 1;
  }
  if (frame.modelMs !== null) {
    totals.modelMs += frame.modelMs;
    totals.modelCalls += 1;
  }
}

export function recordPerfPaint(ageMs: number) {
  if (!PERF_LOGGING) return;
  totals ??= emptyTotals();
  totals.visibleMs += ageMs;
  totals.visibleFrames += 1;
}

export function flushPerfSummary() {
  if (!PERF_LOGGING || !totals || totals.frames === 0) return;
  const elapsed = Math.max(0.001, (performance.now() - totals.startedAt) / 1000);
  const average = (sum: number, count: number) => Math.round(sum / Math.max(1, count));
  const jpeg = average(totals.jpegMs, totals.frames);
  const communication = average(totals.communicationMs, totals.frames);
  const mediapipe = totals.profiledFrames ? average(totals.mediapipeMs, totals.profiledFrames) : null;
  const model = totals.modelCalls ? average(totals.modelMs, totals.modelCalls) : null;
  const screen = totals.visibleFrames ? average(totals.visibleMs, totals.visibleFrames) : null;
  const perFrameModel = totals.modelMs / totals.frames;
  const measured = [
    { name: "JPEG", ms: jpeg },
    { name: "MediaPipe", ms: mediapipe ?? 0 },
    { name: "modelo (média por quadro)", ms: perFrameModel },
    { name: "comunicação/outros", ms: communication },
  ];
  const longest = measured.reduce((a, b) => b.ms > a.ms ? b : a);
  const internal = mediapipe === null ? "pontos e modelo: ative os logs no backend" :
    `pontos/MediaPipe ${mediapipe} ms | modelo ${model === null ? "aguardando 48 quadros" : `${model} ms por predição`}`;
  console.info(
    `[Sinaliza] ${new Date().toLocaleTimeString("pt-BR")} | câmera: ${Math.round(cameraFps)} FPS | análise: ${Math.round(totals.frames / elapsed)} FPS | atraso dos pontos: ${screen === null ? "aguardando" : `${screen} ms`}\n` +
    `  Tempo médio: imagem/JPEG ${jpeg} ms | ${internal} | comunicação/outros ${communication} ms\n` +
    `  Etapa mais lenta medida: ${mediapipe === null ? "ative os logs no backend" : longest.name}`,
  );
  totals = null;
}

export function perfLog(stage: string, fields: Record<string, number | string | boolean | null>) {
  if (PERF_DETAIL) console.info(`[Sinaliza detalhe] ${new Date().toISOString()} ${stage}`, fields);
}
