import type { PredictResponse } from "../services/api";

export interface StabilizationOptions {
  confirmations: number;
  minIntervalMs: number;
}

/** Valores de build opcionais; configuração inválida usa o padrão aprovado. */
export function stabilizationOptions(confirmations?: string, minIntervalMs?: string): StabilizationOptions {
  const count = Number(confirmations?.trim() || 2);
  const interval = Number(minIntervalMs?.trim() || 1500);
  return {
    confirmations: Number.isInteger(count) && count >= 1 ? count : 2,
    minIntervalMs: Number.isFinite(interval) && interval >= 0 ? interval : 1500,
  };
}

/** Aceita palavras atuais, sem fila ou temporizador que publique palpites antigos. */
export class PredictionStabilizer {
  private candidate: string | null = null;
  private count = 0;
  private lastSign: string | null = null;
  private lastAcceptedAt = -Infinity;

  constructor(private options: StabilizationOptions = stabilizationOptions()) {}

  reset() {
    this.candidate = this.lastSign = null;
    this.count = 0;
    this.lastAcceptedAt = -Infinity;
  }

  update(prediction: PredictResponse, nowMs: number): PredictResponse | null {
    // Cooldown não é uma nova classificação e não participa da contagem.
    if (prediction.is_cooldown) return null;
    if (!prediction.sign || (prediction.status && prediction.status !== "recognized")) {
      this.candidate = null;
      this.count = 0;
      return null;
    }
    if (prediction.sign === this.candidate) {
      this.count = Math.min(this.count + 1, this.options.confirmations);
    } else {
      this.candidate = prediction.sign;
      this.count = 1;
    }
    if (this.count < this.options.confirmations || prediction.sign === this.lastSign
        || nowMs - this.lastAcceptedAt < this.options.minIntervalMs) return null;
    this.lastSign = prediction.sign;
    this.lastAcceptedAt = nowMs;
    return prediction;
  }
}
