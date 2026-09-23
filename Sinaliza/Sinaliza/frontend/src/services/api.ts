/**
 * Cliente HTTP para a API Sinaliza.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

/* ---------- Tipos ---------- */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  num_classes: number;
  version: string;
}

export interface PredictionItem {
  class_id: number;
  sign: string;
  confidence: number;
}

export interface PredictResponse {
  sign: string | null;
  confidence: number;
  top_k: PredictionItem[];
  is_cooldown: boolean;
  processing_time_ms: number;
}

export interface DictionaryEntry {
  sign_id: number;
  name: string;
  category: string;
  description: string;
  example_video_url: string | null;
}

export interface DictionaryResponse {
  entries: DictionaryEntry[];
  total: number;
  categories: string[];
}

export interface ModelInfoResponse {
  model_type: string;
  num_classes: number;
  device: string;
  parameters: number;
  smoothing: string;
}

/* ---------- Helpers ---------- */

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/* ---------- Endpoints ---------- */

export const api = {
  health: () => request<HealthResponse>("/health"),

  modelInfo: () => request<ModelInfoResponse>("/model/info"),

  predict: (landmarks: number[][], mask?: boolean[]) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({ landmarks, mask }),
    }),

  resetPredictor: () =>
    request<{ status: string }>("/predict/reset", { method: "POST" }),

  dictionary: (params?: {
    category?: string;
    search?: string;
    skip?: number;
    limit?: number;
  }) => {
    const sp = new URLSearchParams();
    if (params?.category) sp.set("category", params.category);
    if (params?.search) sp.set("search", params.search);
    if (params?.skip !== undefined) sp.set("skip", String(params.skip));
    if (params?.limit !== undefined) sp.set("limit", String(params.limit));
    const qs = sp.toString();
    return request<DictionaryResponse>(`/dictionary${qs ? `?${qs}` : ""}`);
  },
};

export default api;
