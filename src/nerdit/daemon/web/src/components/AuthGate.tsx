import { useEffect, useState, type ReactNode } from "react";
import { Navigate } from "react-router-dom";
import {
  authErrorCode,
  clearStoredToken,
  getStoredToken,
  SESSION_ENDED_EVENT,
  setSessionMode,
  type AuthSession
} from "../lib/auth";
import { Button } from "./ui";

function SessionRecovery({ retry }: { retry: () => void }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="w-full max-w-sm space-y-4">
        <h1 className="text-20 font-semibold">Session ended</h1>
        <p className="text-14 text-muted-foreground">Reopen this machine from Nerdit Cloud.</p>
        <Button onClick={retry}>Try again</Button>
      </div>
    </div>
  );
}

/** The proxy can authenticate with an HttpOnly cookie, without a local token. */
export function AuthGate({ children, login = false }: { children: ReactNode; login?: boolean }) {
  const [state, setState] = useState<"checking" | "ready" | "login" | "expired" | "error">("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const expired = () => setState("expired");
    window.addEventListener(SESSION_ENDED_EVENT, expired);
    return () => window.removeEventListener(SESSION_ENDED_EVENT, expired);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const token = getStoredToken();
    setState("checking");
    void fetch("/api/auth/check", {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: controller.signal,
      cache: "no-store"
    }).then(async (response) => {
      const body = await response.json().catch(() => null);
      if (controller.signal.aborted) return;
      if (response.ok && body?.ok === true && typeof body.role === "string") {
        setSessionMode((body as AuthSession).mode);
        setState("ready");
      } else if ([401, 403].includes(response.status)) {
        clearStoredToken();
        const code = authErrorCode(body);
        setState(code?.startsWith("proxy_session_") ? "expired" : body?.error ? "error" : "login");
      } else {
        setState("error");
      }
    }).catch(() => {
      if (!controller.signal.aborted) setState("error");
    });
    return () => controller.abort();
  }, [attempt, login]);

  if (state === "checking") return <div role="status" className="p-6 text-muted-foreground">Connecting…</div>;
  if (state === "expired") return <SessionRecovery retry={() => setAttempt((value) => value + 1)} />;
  if (state === "error") return (
    <div className="p-6 space-y-4">
      <p role="alert">Unable to connect to this machine.</p>
      <Button onClick={() => setAttempt((value) => value + 1)}>Try again</Button>
    </div>
  );
  if (state === "login") return login ? <>{children}</> : <Navigate to="/login" replace />;
  return login ? <Navigate to="/" replace /> : <>{children}</>;
}
