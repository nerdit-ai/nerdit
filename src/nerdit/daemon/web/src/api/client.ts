import { clearStoredToken, getStoredToken } from "../lib/auth";

export function apiUrl(path: string): string {
  return `/api${path}`;
}

/**
 * Same as a plain `Error` wherever one is caught generically (toasts, etc.)
 * but carries the HTTP status plus the P1 structured-error envelope so a
 * caller can distinguish e.g. 403 (role gate) or 409 (`config.stale`) from
 * other failures, and can show the machine `code`/`hint` without re-parsing.
 *
 * `message` is the envelope `message` (falls back to `detail`/statusText).
 * `code`/`hint` are undefined when the response was not a structured error.
 */
export class ApiError extends Error {
  status: number;
  code?: string;
  hint?: string;

  constructor(message: string, status: number, code?: string, hint?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.hint = hint;
  }
}

/** A response body plus the headers a caller may need (ETag round-trips). */
export interface ApiResult<T> {
  data: T;
  etag: string | null;
  response: Response;
}

/**
 * Core fetch: attaches auth + JSON headers, parses the P1 error envelope on a
 * non-2xx, and returns the parsed body alongside the raw response so a caller
 * can read headers (e.g. ETag for optimistic concurrency).
 */
export async function apiRaw<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const isFormData = init?.body instanceof FormData;
  const headers = new Headers(init?.headers);
  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const token = getStoredToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(apiUrl(path), {
    ...init,
    headers
  });
  if (!response.ok) {
    if (response.status === 401) {
      clearStoredToken();
      if (typeof window !== "undefined") {
        window.location.assign("/login");
      }
    }
    const body = await response.json().catch(() => null);
    throw errorFromEnvelope(body, response);
  }
  const etag = response.headers.get("ETag");
  if (response.status === 204) {
    return { data: undefined as T, etag, response };
  }
  const data = (await response.json()) as T;
  return { data, etag, response };
}

/** Build an `ApiError` from a parsed error body (P1 envelope or legacy `detail`). */
function errorFromEnvelope(body: unknown, response: Response): ApiError {
  const env = (body ?? {}) as {
    code?: string;
    message?: string;
    hint?: string;
    detail?: string;
  };
  const message = env.message ?? env.detail ?? response.statusText;
  return new ApiError(message, response.status, env.code, env.hint);
}

/** Convenience wrapper returning just the parsed body (the common case). */
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const { data } = await apiRaw<T>(path, init);
  return data;
}
