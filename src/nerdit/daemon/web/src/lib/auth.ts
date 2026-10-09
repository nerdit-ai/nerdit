const TOKEN_KEY = "nerdit.token";
const TOKEN_PERSIST_KEY = "nerdit.token.persist";

export type AuthMode = "token" | "local" | "tunnel";
export interface AuthSession {
  ok: true;
  role: string;
  mode?: AuthMode;
}

// Presentation state only. Every API request still authenticates server-side.
let sessionMode: AuthMode | undefined;

export function setSessionMode(mode: AuthMode | undefined): void {
  sessionMode = mode;
}

export const SESSION_ENDED_EVENT = "nerdit:session-ended";

export function handleAuthFailure(): void {
  if (typeof window === "undefined") return;
  if (sessionMode === "tunnel") {
    // A full-page navigation would hit the expired proxy before the daemon
    // could serve its shell. Keep recovery in the already loaded dashboard.
    window.dispatchEvent(new Event(SESSION_ENDED_EVENT));
  } else {
    window.location.assign("/login");
  }
}

/** Daemon errors are flat; the Cloud proxy wraps its detail in `error`. */
export function authErrorCode(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const envelope = body as { error?: { code?: unknown }; code?: unknown };
  const code = envelope.error?.code ?? envelope.code;
  return typeof code === "string" ? code : null;
}

export function getStoredToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY) ?? localStorage.getItem(TOKEN_KEY);
}

export function storeToken(token: string, remember: boolean): void {
  if (remember) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(TOKEN_PERSIST_KEY, "1");
    sessionStorage.removeItem(TOKEN_KEY);
    return;
  }
  sessionStorage.setItem(TOKEN_KEY, token);
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_PERSIST_KEY);
}

export function clearStoredToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_PERSIST_KEY);
}

export function isRememberingToken(): boolean {
  return localStorage.getItem(TOKEN_PERSIST_KEY) === "1";
}

/**
 * Is this non-2xx the daemon saying the BEARER itself is dead (unknown or
 * revoked), rather than "you are authenticated but not an admin"?
 *
 * The distinction cannot be read off the status alone: `ScopedTokenAuthMiddleware`
 * answers an unresolvable token with **403** `invalid_token`, not 401. Treating
 * every 403 as the role gate is how revoking the very token driving an
 * admin-only page left the user wedged: stale token still in storage, "Admin
 * token required" rendered, and any /login visit bounced straight back by the
 * truthy token. Liberal on purpose: a 401 (a proxy, or a future shape) is stale
 * too.
 *
 * It lives here, and not in a page, because BOTH admin-only probes (`Tokens`
 * and `Activity`) must agree — the bug this fixes was one of them not knowing.
 */
export function isDeadBearer(status: number, code: string | null): boolean {
  return status === 401 || (status === 403 && (
    code === "invalid_token" || code === "proxy_session_invalid" || code === "proxy_session_expired"
  ));
}
