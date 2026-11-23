// constants/api.ts
import { ERRORS, getError, type ErrorInfo } from "./errors";

// FastAPI server (uvicorn backend.server:app). Override with VITE_API_URL in .env
export const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Mirrors backend/schemas/classifier.py → Prediction
export interface Prediction {
  label: string;
  analysed_seconds: number;
  processing_ms: number;
}

// Mirrors backend/schemas/classifier.py → Health
export interface Health {
  status: string;
  models: string;
  max_duration: number | null;
}

// GET /api/health. Throws if the server is down, too slow or not "ok".
export async function health(timeoutMs = 5000): Promise<Health> {
  const res = await fetch(`${API_URL}/api/health`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  const body = (await res.json()) as Health;
  if (body.status !== "ok") throw new Error(`model status: ${body.status}`);
  return body;
}

// Everything the result panel can show, one variant at a time
export type ResultState =
  | { status: "empty" }
  | { status: "loading" }
  | { status: "success"; prediction: Prediction }
  | { status: "error"; error: ErrorInfo };

// Thrown by predict() so the caller gets a displayable error
export class ApiError extends Error {
  constructor(public info: ErrorInfo) {
    super(info.message);
  }
}

// Mirrors backend/schemas/classifier.py → ErrorResponse
interface ErrorBody {
  detail?: { code?: string; message?: string };
}

// The backend answers every error with {"detail": {"code", "message"}}; the
// code picks the message. Fall back to the status code if the body is not
// JSON (e.g. a proxy error page).
async function errorFromResponse(res: Response): Promise<ErrorInfo> {
  try {
    const body = (await res.json()) as ErrorBody;
    if (body.detail?.code) return getError(body.detail.code);
  } catch {
    // not JSON
  }
  if (res.status === 413) return ERRORS.FILE_TOO_LARGE;
  if (res.status === 415) return ERRORS.UNSUPPORTED_FORMAT;
  return ERRORS.UNKNOWN_ERROR;
}

export async function predict(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/predict`, { method: "POST", body: form });
  } catch {
    throw new ApiError(ERRORS.NETWORK_ERROR); // server down, CORS, offline
  }

  if (!res.ok) throw new ApiError(await errorFromResponse(res));
  return (await res.json()) as Prediction;
}
